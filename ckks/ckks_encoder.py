"""A module to encode integers as specified in the CKKS scheme.
"""
import numpy as np
from util.ntt import FFTContext
from util.plaintext import Plaintext
from util.polynomial import Polynomial


class CKKSEncoder:
    """An encoder for several complex numbers as specified in the CKKS scheme.
    Attributes:
        degree (int): Degree of polynomial that determines quotient ring.
        fft (FFTContext): FFTContext object to encode/decode.
    """
    def __init__(self, params):
        """Inits CKKSEncoder with the given parameters.
        Args:
            params (Parameters): Parameters including polynomial degree,
                plaintext modulus, and ciphertext modulus.
        """
        self.degree = params.poly_degree
        self.fft = FFTContext(self.degree * 2)
    
    def encode(self, values, scaling_factor):
        """Encodes complex numbers into a polynomial.
        Encodes an array of complex number into a polynomial.
        Args:
            values (list): List of complex numbers to encode.
            scaling_factor (float): Scaling factor to multiply by.
        Returns:
            A Plaintext object which represents the encoded value.
        """
        values_array = np.asarray(values, dtype=complex)
        num_values = len(values_array)
        plain_len = num_values << 1
        
        # Canonical embedding inverse variant.
        to_scale = self.fft.embedding_inv(values)
        to_scale_array = np.asarray(to_scale, dtype=complex)
        
        # Multiply by scaling factor, and split up real and imaginary parts.
        message = np.zeros(plain_len, dtype=int)
        message[:num_values] = np.round(to_scale_array.real * scaling_factor).astype(int)
        message[num_values:] = np.round(to_scale_array.imag * scaling_factor).astype(int)
        
        return Plaintext(Polynomial(plain_len, message.tolist()), scaling_factor)
    
    def decode(self, plain):
        """Decodes a plaintext polynomial.
        Decodes a plaintext polynomial back to a list of integers.
        Args:
            plain (Plaintext): Plaintext to decode.
        Returns:
            A decoded list of integers.
        """
        if not isinstance(plain, Plaintext):
            raise ValueError("Input to decode must be a Plaintext")
        
        coeffs_array = np.asarray(plain.poly.coeffs)
        plain_len = len(coeffs_array)
        num_values = plain_len >> 1
        
        # Divide by scaling factor, and turn back into a complex number.
        real_parts = coeffs_array[:num_values] / plain.scaling_factor
        imag_parts = coeffs_array[num_values:] / plain.scaling_factor
        message = real_parts + 1j * imag_parts
        
        # Compute canonical embedding variant.
        return self.fft.embedding(message.tolist())
