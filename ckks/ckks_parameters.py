"""A module to keep track of parameters for the CKKS scheme."""
import math
import numpy as np
from util.crt import CRTContext


class CKKSParameters:
    """An instance of parameters for the CKKS scheme.
    Attributes:
        poly_degree (int): Degree d of polynomial that determines the
            quotient ring R.
        ciph_modulus (int): Coefficient modulus of ciphertexts.
        big_modulus (int): Large modulus used for bootstrapping.
        scaling_factor (float): Scaling factor to multiply by.
        hamming_weight (int): Hamming weight parameter for sampling secret key.
        taylor_iterations (int): Number of iterations to perform for Taylor series in
            bootstrapping.
        prime_size (int): Minimum number of bits in primes for RNS representation.
        crt_context (CRTContext): Context to manage RNS representation.
    """
    def __init__(self, poly_degree, ciph_modulus, big_modulus, scaling_factor, taylor_iterations=6,
                 prime_size=59):
        """Inits Parameters with the given parameters.
        Args:
            poly_degree (int): Degree d of polynomial of ring R.
            ciph_modulus (int): Coefficient modulus of ciphertexts.
            big_modulus (int): Large modulus used for bootstrapping.
            scaling_factor (float): Scaling factor to multiply by.
            taylor_iterations (int): Number of iterations to perform for Taylor series in
                bootstrapping.
            prime_size (int): Minimum number of bits in primes for RNS representation. Can set to 
                None if using the RNS representation if undesirable.
        """
        self.poly_degree = poly_degree
        self.ciph_modulus = ciph_modulus
        self.big_modulus = big_modulus
        self.scaling_factor = scaling_factor
        self.num_taylor_iterations = taylor_iterations
        self.hamming_weight = poly_degree // 4
        self.crt_context = None
        if prime_size:
            num_primes = 1 + int((1 + np.log2(poly_degree) + 4 * np.log2(big_modulus)) / prime_size)
            self.crt_context = CRTContext(num_primes, prime_size, poly_degree)
    
    def print_parameters(self):
        """Prints parameters.
        """
        print("Encryption parameters")
        print("\t Polynomial degree: %d" %(self.poly_degree))
        print("\t Ciphertext modulus size: %d bits" % (int(np.log2(self.ciph_modulus))))
        print("\t Big ciphertext modulus size: %d bits" % (int(np.log2(self.big_modulus))))
        print("\t Scaling factor size: %d bits" % (int(np.log2(self.scaling_factor))))
        print("\t Number of Taylor iterations: %d" % (self.num_taylor_iterations))
        if self.crt_context:
            rns = "Yes"
        else:
            rns = "No"
        print("\t RNS: %s" % (rns))
