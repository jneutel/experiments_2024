cfm_to_cms = 0.00047194745  # cubic feet per minute to cubic meter per second

# at 300k according to https://www.ohio.edu/mechanical/thermo/property_tables/air/air_Cp_Cv.html
cp_air = 1.005  # kJ/kg.K
latent_heat_condensation = 2501  # kJ/kg

# At 101.325 kPa (abs) and 20C (68F) according to https://en.wikipedia.org/wiki/Density_of_air
rho_air = 1.2041  # kg/m3
delta_F_to_K = 5 / 9
kW_to_ton = 0.284345

gpm_to_cms = 6.30902e-5  # m3/gpm
rho_water = 997  # kg/m3
cp_water = 4.179  # kJ / kg.K

ton_to_kw_chiller = 0.6  # kW / ton
sqft_to_sqm = 0.092903  # m^2/ft^2

# Energy conversions
kWh_to_MJ = 3.6
tonhr_to_MJ = 1.0 / 0.07898476
mmbtu_to_TJ = 1.05506 * 1e-3
mega_to_tera = 1.0e-6
kilo_to_mega = 1e-3
tonhr_to_kwh = tonhr_to_MJ / kWh_to_MJ
tonhr_to_TJ = tonhr_to_MJ * mega_to_tera
kwh_to_TJ = kWh_to_MJ * mega_to_tera
TJ_to_MWh = 1 / kwh_to_TJ * kilo_to_mega

# Area conversions
sqfeet_to_sqmeter = 0.092903


def cop_to_kW_per_ton(cop):
    return 1 / cop / kW_to_ton


# Temperature conversions
def F_to_C(F):
    if isinstance(F, list):
        return [(x - 32) * (5 / 9) for x in F]
    else:
        return (F - 32) * (5 / 9)
