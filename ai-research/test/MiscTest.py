import torch

if __name__ == '__main__':
    # A = torch.tensor([[1, 2,3],
    #                   [4,5, 6]])
    #
    # B = torch.tensor([[7, 8],
    #                   [9, 10],
    #                   [11, 12]])
    # print(A.shape,B.shape)
    # print(f'A*b=:{A@B},torch.mutmul:{torch.matmul(A,B)}')
    # print(A@B == torch.matmul(A,B))
    '''
        1*7+2*9+3*11=58     1*8+2*10+3*12=64
        4*7+5*9+6*11=139    4*8+5*10+6*12=
        28+45+66             32+50+72
    '''

    # x = 3
    # y = 4
    # A = torch.randn(x, y)
    # B = torch.randn(x, y)
    #
    # print(A * B == torch.mul(A, B))
    batch = 2
    # a = torch.tensor([2, 2, 5])
    # b = torch.tensor([2, 5, 6])
    a = torch.randn(2, 3, 4)
    b = torch.randn( 4, 5)
    c1 = torch.matmul(a, b)
    c2 = torch.bmm(a, b.unsqueeze(0).expand(batch,-1,-1))
    print(c1.shape == c2.shape)