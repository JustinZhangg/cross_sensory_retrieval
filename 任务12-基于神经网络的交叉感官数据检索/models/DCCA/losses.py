import torch

class cca_loss():
    def __init__(self, outdim_size, use_all_singular_values, device):
        self.outdim_size = outdim_size
        self.use_all_singular_values = use_all_singular_values
        self.device = device

    def __call__(self, H1, H2):
        r1 = 1e-3
        r2 = 1e-3
        eps = 1e-9

        H1, H2 = H1.t(), H2.t()

        o1 = o2 = H1.size(0)

        m = H1.size(1)

        H1_bar = H1 - H1.mean(dim=1).unsqueeze(dim=1)
        H2_bar = H2 - H2.mean(dim=1).unsqueeze(dim=1)

        SigmaHat12 = (1.0 / (m - 1)) * torch.matmul(H1_bar, H2_bar.t())
        SigmaHat11 = (1.0 / (m - 1)) * torch.matmul(H1_bar, H1_bar.t()) + r1 * torch.eye(o1).to(self.device)
        SigmaHat22 = (1.0 / (m - 1)) * torch.matmul(H2_bar, H2_bar.t()) + r2 * torch.eye(o2).to(self.device)

        # Calculating the root inverse of covariance matrices by using eigen decomposition
        [D1, V1] = torch.symeig(SigmaHat11, eigenvectors=True)
        [D2, V2] = torch.symeig(SigmaHat22, eigenvectors=True)

        # Filtering out negative and small eigenvalues
        posInd1 = torch.gt(D1, eps).nonzero()[:, 0]
        D1 = D1[posInd1]
        V1 = V1[:, posInd1]
        posInd2 = torch.gt(D2, eps).nonzero()[:, 0]
        D2 = D2[posInd2]
        V2 = V2[:, posInd2]

        SigmaHat11RootInv = torch.matmul(
            torch.matmul(V1, torch.diag(D1 ** -0.5)), V1.t())
        SigmaHat22RootInv = torch.matmul(
            torch.matmul(V2, torch.diag(D2 ** -0.5)), V2.t())

        Tval = torch.matmul(torch.matmul(SigmaHat11RootInv,
                                         SigmaHat12), SigmaHat22RootInv)

        if self.use_all_singular_values:
            # all singular values are used to calculate the correlation
            tmp = torch.matmul(Tval.t(), Tval)
            corr = torch.trace(torch.sqrt(tmp))
        else:
            # just the top self.outdim_size singular values are used
            trace_TT = torch.matmul(Tval.t(), Tval)
            trace_TT = torch.add(trace_TT, r1*torch.eye(trace_TT.shape[0]).to(H1.device))
            U, _ = torch.symeig(trace_TT, eigenvectors=True)
            U = U.topk(min(U.size(0), self.outdim_size))[0]
            corr = torch.trace(torch.sqrt(U))
            
        return -corr