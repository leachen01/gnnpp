import torch
import numpy as np


def crps_no_avg(mu_sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Calculates the Continuous Ranked Probability Score (CRPS) assuming normally distributed df

    :param torch.Tensor mu_sigma: tensor of mean and standard deviation
    :param torch.Tensor y: observed df

    :return tensor: CRPS value
    :rtype torch.Tensor
    """
    mu, sigma = torch.split(mu_sigma, 1, dim=-1)
    y = y.view((-1, 1))  # make sure y has the right shape
    pi = np.pi  # 3.14159265359
    omega = (y - mu) / sigma
    # PDF of normal distribution at omega
    pdf = 1 / (torch.sqrt(torch.tensor(2 * pi))) * torch.exp(-0.5 * omega**2)

    # Source:
    # https://stats.stackexchange.com/questions/187828/how-are-the-error-function-and-standard-normal-distribution-function-related
    cdf = 0.5 * (1 + torch.erf(omega / torch.sqrt(torch.tensor(2))))

    crps_score = sigma * (omega * (2 * cdf - 1) + 2 * pdf - 1 / torch.sqrt(torch.tensor(pi)))
    return crps_score


def crps_active_stations(mu_sigma: torch.Tensor, y: torch.Tensor, active_stations: torch.Tensor) -> torch.Tensor:
    """Calculates the Continuous Ranked Probability Score (CRPS) for all stations which have valid measurements

    :param torch.Tensor mu_sigma: tensor of mean and standard deviation
    :param torch.Tensor y: observed df
    :param torch.Tensor active_stations: tensor of active stations

    :return tensor: CRPS value
    :rtype torch.Tensor
    """
    active_stations = active_stations.to(torch.bool)
    active_stations = ~active_stations

    mu_sigma = mu_sigma[active_stations]
    y = y[active_stations]
    crps_score = crps_averaged(mu_sigma=mu_sigma, y=y)
    return crps_score


def crps_averaged(mu_sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Calculates the Continuous Ranked Probability Score (CRPS) assuming normally distributed df

    :param torch.Tensor mu_sigma: tensor of mean and standard deviation
    :param torch.Tensor y: observed df

    :return tensor: CRPS value
    :rtype torch.Tensor
    """
    crps_score = crps_no_avg(mu_sigma=mu_sigma, y=y)
    return torch.mean(crps_score)


class NormalCRPS(torch.nn.Module):
    """Source: Höhlein et. al (2024) Postprocessing of Ensemble Weather Forecasts Using
    Permutation-Invariant Neural Networks
    https://github.com/khoehlein/Permutation-invariant-Postprocessing/blob/main/model/loss/losses.py
    """

    def __init__(self):
        super(NormalCRPS, self).__init__()
        self._inv_sqrt_pi = 1 / torch.sqrt(torch.tensor(np.pi))
        self.dist = torch.distributions.Normal(loc=0.0, scale=1.0)

    def crps(self, mu_sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Calculates the Continuous Ranked Probability Score (CRPS) assuming normally distributed df

        :param torch.Tensor mu_sigma: tensor of mean and standard deviation
        :param torch.Tensor y: observed df

        :return tensor: CRPS value
        :rtype torch.Tensor
        """
        mask = ~torch.isnan(y)
        mu, sigma = torch.split(mu_sigma, 1, dim=1)
        y = y.unsqueeze(1)

        mu = mu[mask]
        sigma = sigma[mask]
        y = y[mask]

        z_red = (y - mu) / sigma

        cdf = self.dist.cdf(z_red)
        pdf = torch.exp(self.dist.log_prob(z_red))
        crps = sigma * (z_red * (2.0 * cdf - 1.0) + 2.0 * pdf - self._inv_sqrt_pi)
        crps_score = torch.mean(crps)
        return crps_score
