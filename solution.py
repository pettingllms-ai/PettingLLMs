
import sys

def solve() -> None:
    data = sys.stdin.read().strip().split()
    if not data:
        return
    
    s = data[0]
    x = int(data[1])
    
    n = len(s)
    
    for k in range(0, n + 1):
        if k > 0 and '0' in s[:k]:
            continue
        
        if k < n and '0' in s[k:]:
            continue
        
        a_val = int(s[:k]) if k > 0 else 0
        b_val = int(s[k:]) if k < n else 0
        
        if a_val + b_val == x:
            if k > 0:
                print(f"1 {k}")
            else:
                print(f"1 0")
            
            if k < n:
                print(f"{k+1} {n}")
            else:
                print(f"{n+1} 0")
            return
    
    raise RuntimeError("Solution not found")

if __name__ == "__main__":
    solve()
