/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 11, 2022.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
/**
 *
 */

/*! @mainpage Heimdal crypto library
 *
 * @section intro Introduction
 *
 * Heimdal libhcrypto library is a implementation many crypto
 * algorithms, among others: AES, SHA, DES, RSA, Camellia and many
 * help function.
 *
 * hcrypto provies a OpenSSL compatible interface libcrypto interface
 * and is licensed under a 3 clause BSD license (GPL compatible).
 *
 * The project web page: http://www.h5l.org/
 *
 * Sections of this manual:
 *
 * - @subpage page_evp, @ref hcrypto_evp
 * - @subpage page_rand, @ref hcrypto_rand
 * - @subpage page_dh, @ref hcrypto_dh
 * - @subpage page_rsa, @ref hcrypto_rsa
 * - @ref hcrypto_misc
 *
 * Older interfaces that you should not use:
 *
 * - @subpage page_des, @ref hcrypto_des
 *
 * @subsection control_functions Control functions
 *
 * Functions controlling general behavior, like adding algorithms, are
 * documented in this module: @ref hcrypto_core .
 *
 * @subsection return_values Return values
 *
 * Return values are diffrent in this module to be compatible with
 * OpenSSL interface. The diffrence is that on success 1 is returned
 * instead of the customary 0.

 * @subsection History
 *
 * Eric Young implemented DES in the library libdes, that grew into
 * libcrypto in the ssleay package. ssleay went into recession and
 * then got picked up by the OpenSSL (htp://www.openssl.org/)
 * project.
 *
 * libhcrypto is an independent implementation with no code decended
 * from ssleay/openssl. Both includes some common imported code, for
 * example the AES implementation.
 */

/** @defgroup hcrypto_dh Diffie-Hellman functions
 * See the @ref page_dh for description and examples.
 */
/** @defgroup hcrypto_rsa RSA functions
 * See the @ref page_rsa for description and examples.
 */
/** @defgroup hcrypto_evp EVP generic crypto functions
 * See the @ref page_evp for description and examples.
 */
/** @defgroup hcrypto_rand RAND crypto functions
 * See the @ref page_rand for description and examples.
 */
/** @defgroup hcrypto_des DES crypto functions
 * See the @ref page_des for description and examples.
 */
/** @defgroup hcrypto_core hcrypto function controlling behavior */
/** @defgroup hcrypto_misc hcrypto miscellaneous functions */
