import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt

"""WKB-instantaneous basis quench test: tanh-step vs erf-step.

Implements the conventions you specified:

- Freeze scale factor during quench: a(eta) ≈ a0.
- Instantaneous WKB frequency (conformal coupling, frozen-a approximation):
    omega_k(eta) = sqrt(k^2 + a0^2 m^2(eta)).
- Bogoliubov decomposition in WKB basis:
    chi_k(eta) = alpha_k(eta) f_k^WKB(eta) + beta_k(eta) f_k^{WKB*}(eta)
  with evolution equations:
    Phi'   = omega
    alpha' = (omega'/(2 omega)) * exp(+2 i Phi) * beta
    beta'  = (omega'/(2 omega)) * exp(-2 i Phi) * alpha
- Complexity:
    K = ∫ dη ∫_0^∞ dk k^2/(2π^2) |beta'(η)|^2
  We compute K exactly by accumulating ∫|beta'|^2 dη per k while integrating.
- Entropy density:
    s(n) = (n+1) ln(n+1) - n ln n.
- Horizon-loss at evaluation time t=H^{-1}:
    a_eval = e,  k_hor = a_eval * H = e (in H=1 units).

Note: This is a *falsification-kernel* script, not a production-grade numerics stack.
It trades speed for transparency.
"""

# Precision
mp.mp.dps = 70

# Global parameters (Hubble units)
H = mp.mpf('1')
a0 = mp.mpf('1')
mf = mp.mpf('1')
eta0 = mp.mpf('0')

# Quench integration window: [eta0 - L*sigma, eta0 + L*sigma]
L_window = mp.mpf('12')

# Evaluation time for Delta S_loss: t=H^{-1} => a=e (H=1)
a_eval = mp.e
k_hor = a_eval * H

# -------------
# Profiles
# -------------

def m2_tanh(eta, sigma):
    # m^2(eta) = mf^2 * (1 + tanh((eta-eta0)/sigma))/2
    x = (eta - eta0) / sigma
    return mf**2 * (1 + mp.tanh(x)) / 2

def dm2_tanh(eta, sigma):
    x = (eta - eta0) / sigma
    return mf**2 * (mp.sech(x)**2) / (2 * sigma)

def m2_erf(eta, sigma):
    # m^2(eta) = mf^2 * (1 + erf((eta-eta0)/sigma))/2
    x = (eta - eta0) / sigma
    return mf**2 * (1 + mp.erf(x)) / 2

def dm2_erf(eta, sigma):
    x = (eta - eta0) / sigma
    return mf**2 * mp.e**(-x**2) / (mp.sqrt(mp.pi) * sigma)

# -------------
# Entropy density
# -------------

def s_of_n(n):
    n = mp.mpf(n)
    if n <= 0:
        return mp.mpf('0')
    # for tiny n, use series-stable form
    if n < mp.mpf('1e-30'):
        return n * (1 - mp.log(n))
    return (n + 1) * mp.log(n + 1) - n * mp.log(n)

# -------------
# Bogoliubov ODE integrator (RK4)
# -------------

def evolve_k(k, sigma, profile='tanh', nsteps=5000):
    """Return (n_k, K_k) for a single k.

    n_k = |beta(eta_out)|^2
    K_k = ∫_{eta_in}^{eta_out} |beta'(eta)|^2 dη

    profile: 'tanh' or 'erf'
    """
    k = mp.mpf(k)
    sigma = mp.mpf(sigma)

    if profile == 'tanh':
        m2 = m2_tanh
        dm2 = dm2_tanh
    elif profile == 'erf':
        m2 = m2_erf
        dm2 = dm2_erf
    else:
        raise ValueError("profile must be 'tanh' or 'erf'")

    eta_in = eta0 - L_window * sigma
    eta_out = eta0 + L_window * sigma
    h = (eta_out - eta_in) / nsteps

    # state variables
    alpha = mp.mpc(1)
    beta = mp.mpc(0)
    Phi = mp.mpf('0')

    # helper functions
    def omega(eta):
        return mp.sqrt(k**2 + (a0**2) * m2(eta, sigma))

    def omegap(eta):
        Om = omega(eta)
        # omega' = (a0^2 m2')/(2 omega)
        return (a0**2) * dm2(eta, sigma) / (2 * Om)

    def deriv(eta, alpha, beta, Phi):
        Om = omega(eta)
        Omp = omegap(eta)
        c = Omp / (2 * Om)
        e2p = mp.e**(2j * Phi)
        e2m = mp.e**(-2j * Phi)
        dalpha = c * e2p * beta
        dbeta = c * e2m * alpha
        dPhi = Om
        return dalpha, dbeta, dPhi

    # accumulate K_k
    Kk = mp.mpf('0')

    eta = eta_in
    for _ in range(nsteps):
        # accumulate |beta'|^2 at the left endpoint (sufficient at high steps)
        Om = omega(eta)
        Omp = omegap(eta)
        c = Omp / (2 * Om)
        dbeta = c * mp.e**(-2j * Phi) * alpha
        Kk += (dbeta.real**2 + dbeta.imag**2) * h

        # RK4 step
        k1a, k1b, k1p = deriv(eta, alpha, beta, Phi)
        k2a, k2b, k2p = deriv(eta + h/2, alpha + h*k1a/2, beta + h*k1b/2, Phi + h*k1p/2)
        k3a, k3b, k3p = deriv(eta + h/2, alpha + h*k2a/2, beta + h*k2b/2, Phi + h*k2p/2)
        k4a, k4b, k4p = deriv(eta + h, alpha + h*k3a, beta + h*k3b, Phi + h*k3p)

        alpha = alpha + (h/6) * (k1a + 2*k2a + 2*k3a + k4a)
        beta = beta + (h/6) * (k1b + 2*k2b + 2*k3b + k4b)
        Phi = Phi + (h/6) * (k1p + 2*k2p + 2*k3p + k4p)
        eta = eta + h

    nk = beta.real**2 + beta.imag**2
    return nk, Kk

# -------------
# k-integrals with cached grid (log-k)
# -------------

def compute_all_for_sigma(sigma, profile='tanh',
                          kmin=mp.mpf('1e-6'), kmax=mp.mpf('80'), Nk=220,
                          nsteps=5000):
    """Compute (R_op, K, S_e, S_loss) for a given sigma and profile."""

    sigma = mp.mpf(sigma)

    # log grid
    xs = np.linspace(float(mp.log(kmin)), float(mp.log(kmax)), Nk)
    ks = [mp.e**mp.mpf(x) for x in xs]

    nk_list = []
    Kk_list = []
    for k in ks:
        nk, Kk = evolve_k(k, sigma, profile=profile, nsteps=nsteps)
        nk_list.append(nk)
        Kk_list.append(Kk)

    # Simpson in log-k: ∫ f(k) dk = ∫ f(e^x) e^x dx
    dx = mp.mpf(xs[1] - xs[0])

    def simpson_from_vals(vals):
        # vals are sampled on uniform x-grid
        N = len(vals)
        if N < 3:
            return mp.mpf('0')
        if N % 2 == 0:
            # make it odd by dropping last point
            N = N - 1
            vals = vals[:N]
        S = vals[0] + vals[-1]
        S += 4 * sum(vals[1:-1:2])
        S += 2 * sum(vals[2:-2:2])
        return S * dx / 3

    # Build integrands on x-grid
    K_vals = []
    Se_vals = []
    Sl_vals = []

    for k, nk, Kk in zip(ks, nk_list, Kk_list):
        # measure factor k^2/(2π^2)
        w = (k**2) / (2 * mp.pi**2)

        # K integrand: w * Kk
        K_vals.append(w * Kk * k)  # *dk/dx = k

        # S_e integrand: w * s(n)
        Se_vals.append(w * s_of_n(nk) * k)

        # Delta S_loss integrand: restrict to k <= k_hor
        if k <= k_hor:
            Sl_vals.append(w * s_of_n(nk) * k)
        else:
            Sl_vals.append(mp.mpf('0'))

    K = simpson_from_vals(K_vals)
    S_e = simpson_from_vals(Se_vals)
    S_loss = simpson_from_vals(Sl_vals)

    denom = S_e + S_loss
    R_op = K / denom if denom != 0 else mp.mpf('0')

    return R_op, K, S_e, S_loss

# -------------
# Analytic sanity checks (your weak-production constant-omega approximation)
# -------------

def K_tanh_approx(sigma):
    # in H=1, a0=1, mf=1: 1/(384 π σ)
    return mp.mpf('1') / (384 * mp.pi * mp.mpf(sigma))

def K_erf_approx(sigma):
    # ratio erf/tanh ≈ 3 / sqrt(2π) if omega treated constant in η
    return (3 / mp.sqrt(2 * mp.pi)) * K_tanh_approx(sigma)

# -------------
# Main sweep
# -------------

def main():
    sigmas = np.linspace(0.10, 0.40, 13)

    results = { 'tanh': [], 'erf': [] }

    # Tunables: increasing these improves accuracy but costs time
    Nk = 180
    kmax = mp.mpf('80')
    nsteps = 4500

    print("Running WKB-basis quench test (frozen a=a0=1).")
    print(f"Evaluation: t=H^-1 => k_hor = e ≈ {mp.nstr(k_hor, 8)}")
    print(f"k-grid: Nk={Nk}, k in [{mp.nstr(mp.mpf('1e-6'), 3)}, {mp.nstr(kmax, 4)}], RK steps per k: {nsteps}")

    for sigma in sigmas:
        sigma_mp = mp.mpf(str(sigma))

        for prof in ['tanh','erf']:
            R_op, K, S_e, S_loss = compute_all_for_sigma(
                sigma_mp,
                profile=prof,
                kmin=mp.mpf('1e-6'),
                kmax=kmax,
                Nk=Nk,
                nsteps=nsteps,
            )
            results[prof].append((float(R_op), float(K), float(S_e), float(S_loss)))

        # Quick K sanity check for tanh
        K_num = mp.mpf(results['tanh'][-1][1])
        K_apx = K_tanh_approx(sigma_mp)
        print(f"sigma={sigma:.3f}: K_tanh(num)={mp.nstr(K_num, 6)}  K_tanh(approx)={mp.nstr(K_apx, 6)}")

    # Plot R_op for both profiles
    R_star = np.sqrt(3)
    plt.figure(figsize=(10, 6))
    plt.axhline(R_star, color='red', linestyle='--', linewidth=1.5, label=r'$R^*=\sqrt{3}$')

    for prof, color in [('tanh','navy'), ('erf','darkgreen')]:
        Rops = np.array([r[0] for r in results[prof]])
        plt.plot(sigmas, Rops, marker='o', linewidth=2, color=color, label=f"{prof} (WKB)")

    plt.xlabel(r'Quench width $\sigma$ [Hubble units]')
    plt.ylabel(r'$R_{\rm op}=K/(S_e+\Delta S_{\rm loss})$')
    plt.title('WKB-basis Op-Complexity Ratio vs Quench Width')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig('/mnt/data/R_op_wkb_erf_vs_tanh.png', dpi=200)
    plt.show()

    print("Saved plot: /mnt/data/R_op_wkb_erf_vs_tanh.png")

if __name__ == '__main__':
    main()
