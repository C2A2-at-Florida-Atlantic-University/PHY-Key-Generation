import numpy as np
# import unireedsolomon as rs
import reedsolo as rs
from tqdm import tqdm

class BitToSymbolTransformation:
    def __init__(self):
        pass
    
    def arr2str(self, arr):
        str_arr = ''
        for i in arr:
            str_arr += str(i)
        return str_arr
    
    def bin_to_2bps(self, A):
        symbols = [1,2,3,4]
        A = self.arr2str(A)
        A_symbols = [symbols[int(A[i:i+2], 2)] for i in range(0, len(A), 2)]
        A_symbols = self.arr2str(A_symbols)
        return A_symbols
    
    def bin_to_hex(self, A):
        A = hex(int(self.arr2str(A), 2))
        A = str(A[2:]) #A hex key
        return A
    
    def bin_to_bytes(self, A):
        A_bits = self.arr2str(A) if isinstance(A, (list, np.ndarray)) else str(A)
        num_bytes = (len(A_bits) + 7) // 8
        A_bytes = int(A_bits, 2).to_bytes(num_bytes, 'big')
        A_str = A_bytes.decode('latin-1')
        return A_str
    
    def group_b2s(self, arr, bits_per_symbol):
        out = []
        for i in range(0, len(arr)):
            if bits_per_symbol == 2:
                out.append(self.bin_to_2bps(arr[i]))
            elif bits_per_symbol == 4:
                out.append(self.bin_to_hex(arr[i]))
            elif bits_per_symbol == 8:
                out.append(self.bin_to_bytes(arr[i]))
            else:
                raise ValueError("Unsupported bits_per_symbol: ", bits_per_symbol)
        return out

class ReedSolomonReconciliation:
    def __init__(self, L, bits_per_symbol, K, N):
        self.L = L
        self.bits_per_symbol = bits_per_symbol
        self.K = K
        self.N = N
        self.S = N - K
        # RS over GF(256) has maximum nsize 255. Ensure parity S <= 254 and k_per_block >= 1.
        if self.S >= 255:
            original_S = self.S
            self.S = 254
            self.N = self.K + self.S
            print(f"Clamping S from {original_S} to {self.S} to satisfy GF(256) constraints (n<=255)")
    
    def _ensure_bytes(self, data):
        if isinstance(data, (bytes, bytearray)):
            return bytes(data)
        if isinstance(data, str):
            return data.encode('latin-1', errors='strict')
        if isinstance(data, np.ndarray):
            data = data.tolist()
        if isinstance(data, list):
            # If it's a list of bits (0/1), pack into bytes; otherwise treat as byte values
            if all(isinstance(x, (int, np.integer)) and x in (0, 1) for x in data):
                bit_str = ''.join(str(int(b)) for b in data)
                if len(bit_str) == 0:
                    return b''
                num_bytes = (len(bit_str) + 7) // 8
                return int(bit_str, 2).to_bytes(num_bytes, 'big')
            else:
                return bytes(int(x) & 0xFF for x in data)
        raise TypeError(f"Unsupported data type for RS encoding: {type(data)}")

    def reconcile(self, A, B):
        # print("Reconciliation with S: ", self.S, ", N: ", self.N, ", K: ", self.K)
        A_bytes = self._ensure_bytes(A)
        B_bytes = self._ensure_bytes(B)

        # RS over GF(256) supports max nsize = 255. If requested N > 255, do block-wise reconciliation.
        max_nsize = 255
        max_k_per_block = max_nsize - self.S
        if self.N <= max_nsize and len(A_bytes) <= (max_nsize - self.S):
            # Single-block case
            nsize = self.S + len(A_bytes)
            rsc = rs.RSCodec(self.S, nsize=nsize)
            Aencode = rsc.encode(A_bytes)
            Aparity = Aencode[len(A_bytes):]
            BAparity = B_bytes + Aparity
            try:
                rmes, rmesecc, errata_pos = rsc.decode(BAparity)
                Breconciled_bytes = rmes
            except Exception as e:
                # print("Reconciliation failed with error: ", e)
                Breconciled_bytes = B_bytes
        else:
            # Multi-block case
            if max_k_per_block <= 0:
                raise ValueError("Invalid parameters: parity S too large for GF(256)")
            Breconciled_parts = []
            start = 0
            while start < len(A_bytes):
                end = min(start + max_k_per_block, len(A_bytes))
                A_block = A_bytes[start:end]
                B_block = B_bytes[start:end]
                nsize_block = self.S + len(A_block)
                if nsize_block > max_nsize:
                    # Adjust block to respect nsize limit strictly
                    end = start + (max_nsize - self.S)
                    A_block = A_bytes[start:end]
                    B_block = B_bytes[start:end]
                    nsize_block = self.S + len(A_block)
                rsc = rs.RSCodec(self.S, nsize=nsize_block)
                Aencode = rsc.encode(A_block)
                Aparity = Aencode[len(A_block):]
                BAparity = B_block + Aparity
                try:
                    rmes, rmesecc, errata_pos = rsc.decode(BAparity)
                    Breconciled_parts.append(rmes)
                except Exception as e:
                    # print(f"Block {start}:{end} reconciliation failed with error: ", e)
                    Breconciled_parts.append(B_block)
                start = end
            Breconciled_bytes = b"".join(Breconciled_parts)

        # Convert to strings for consistency with caller expectations
        Breconciled = Breconciled_bytes.decode('latin-1')
        A_str = A_bytes.decode('latin-1')
        return [A_str == Breconciled, Breconciled]
    
    def reconcile_rate(self, data):
        j = 0
        reconciliation_data1 = []
        reconciliation_data2 = []
        reconciliation_data3 = []
        reconciled_data = []
        pbar = tqdm(total = len(data)/4+1)
        while j <= len(data)-3:
            reconciliation_data1.append(self.reconcile(data[j],data[j+2]))
            reconciliation_data2.append(self.reconcile(data[j],data[j+1]))
            reconciliation_data3.append(self.reconcile(data[j+2],data[j+3]))
            j = j + 4
            pbar.update(1)
            
        return reconciliation_data1, reconciliation_data2, reconciliation_data3

if __name__ == "__main__":
    L = 128
    bits_per_symbol = 2
    K = int(L/bits_per_symbol)
    # S = int(K/128 - 1) 
    maxS = K-1 if 2**8-2 >= K else 2**8-2
    minS = 2**1-1
    midS = 2**4-1
    print("maxS: ", maxS)
    print("minS: ", minS)
    print("midS: ", midS)
    N = K+maxS
    S = N - K
    reconciliation = ReedSolomonReconciliation(L, bits_per_symbol, K, N)
    b2s = BitToSymbolTransformation()
    print("RS parameters:")
    print("L: ", L)
    print("bits_per_symbol: ", bits_per_symbol)
    print("K (L/bits_per_symbol): ", K)
    print("N: ", N)
    print("S (N-K): ", S)
    
    # Generate a random binary key of length L
    keyA = np.random.randint(0, 2, L)
    print("Key A: ", keyA, "Length: ", len(keyA))
    # Number of errors to introduce
    max_errors = int((S/2)*bits_per_symbol)
    additional_errors = 10
    additional_correct = max_errors
    num_errors = int(max_errors+additional_errors-additional_correct)
    # flip the first num_errors bits of key B using the flipped_key
    flipped_key = 1 - keyA
    # randomly select the bits to flip
    israndom = True
    if israndom:
        bits_to_flip = np.random.choice(L, num_errors, replace=False)
        print("Bits to flip: ", bits_to_flip)
        keyB = keyA.copy()
        keyB[bits_to_flip] = flipped_key[bits_to_flip]
    else:
        keyB = np.concatenate((flipped_key[0:num_errors], keyA[num_errors:]))
    print("Key B: ", keyB, "Length: ", len(keyB))
    num_bit_errors = 0
    for i in range(len(keyA)):
        if keyA[i] != keyB[i]:
            num_bit_errors += 1
    print("Num bit errors: ", num_bit_errors)
    if bits_per_symbol == 1:
        keyA = b2s.arr2str(keyA)
        keyB = b2s.arr2str(keyB)
    elif bits_per_symbol == 2:
        keyA, keyB = b2s.bin_to_2bps(keyA), b2s.bin_to_2bps(keyB)
    elif bits_per_symbol == 4:
        keyA, keyB = b2s.bin_to_hex(keyA), b2s.bin_to_hex(keyB)
    elif bits_per_symbol == 8:
        keyA, keyB = b2s.bin_to_bytes(keyA), b2s.bin_to_bytes(keyB)
    else:
        raise ValueError("Invalid bits_per_symbol")
    print("Key A: ", keyA, "Length: ", len(keyA))
    print("Key B: ", keyB, "Length: ", len(keyB))
    # check for number of symbol errors
    num_symbol_errors = 0
    for i in range(len(keyA)):
        if keyA[i] != keyB[i]:
            num_symbol_errors += 1
    print("Num symbol errors: ", num_symbol_errors)
    result, reconciled_key = reconciliation.reconcile(keyA, keyB)
    print("Reconciliation result: ", result)
    print("Reconciled key: ", reconciled_key)