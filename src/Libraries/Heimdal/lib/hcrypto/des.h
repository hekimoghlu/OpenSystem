/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 19, 2024.
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
/* $Id$ */

#ifndef _DESperate_H
#define _DESperate_H 1

/* symbol renaming */
#define	_DES_ipfp_test _hc_DES_ipfp_test
#define DES_cbc_cksum hc_DES_cbc_cksum
#define DES_cbc_encrypt hc_DES_cbc_encrypt
#define DES_cfb64_encrypt hc_DES_cfb64_encrypt
#define DES_check_key_parity hc_DES_check_key_parity
#define DES_ecb3_encrypt hc_DES_ecb3_encrypt
#define DES_ecb_encrypt hc_DES_ecb_encrypt
#define DES_ede3_cbc_encrypt hc_DES_ede3_cbc_encrypt
#define DES_encrypt hc_DES_encrypt
#define DES_generate_random_block hc_DES_generate_random_block
#define DES_init_random_number_generator hc_DES_init_random_number_generator
#define DES_is_weak_key hc_DES_is_weak_key
#define DES_key_sched hc_DES_key_sched
#define DES_new_random_key hc_DES_new_random_key
#define DES_pcbc_encrypt hc_DES_pcbc_encrypt
#define DES_rand_data hc_DES_rand_data
#define DES_random_key hc_DES_random_key
#define DES_read_password hc_DES_read_password
#define DES_set_key hc_DES_set_key
#define DES_set_key_checked hc_DES_set_key_checked
#define DES_set_key_unchecked hc_DES_set_key_unchecked
#define DES_set_key_sched hc_DES_set_key_sched
#define DES_set_odd_parity hc_DES_set_odd_parity
#define DES_set_random_generator_seed hc_DES_set_random_generator_seed
#define DES_set_sequence_number hc_DES_set_sequence_number
#define DES_string_to_key hc_DES_string_to_key

/*
 *
 */

#define DES_CBLOCK_LEN 8
#define DES_KEY_SZ 8

#define DES_ENCRYPT 1
#define DES_DECRYPT 0

typedef unsigned char DES_cblock[DES_CBLOCK_LEN];
typedef struct DES_key_schedule
{
	uint32_t ks[32];
} DES_key_schedule;

/*
 *
 */

#ifndef HC_DEPRECATED
#if defined(__GNUC__) && ((__GNUC__ > 3) || ((__GNUC__ == 3) && (__GNUC_MINOR__ >= 1 )))
#define HC_DEPRECATED __attribute__((deprecated))
#elif defined(_MSC_VER) && (_MSC_VER>1200)
#define HC_DEPRECATED __declspec(deprecated)
#else
#define HC_DEPRECATED
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

void	DES_set_odd_parity(DES_cblock *);
int	DES_check_key_parity(DES_cblock *);
int	DES_is_weak_key(DES_cblock *);
int	HC_DEPRECATED DES_set_key(DES_cblock *, DES_key_schedule *);
int	DES_set_key_checked(DES_cblock *, DES_key_schedule *);
int	DES_set_key_unchecked(DES_cblock *, DES_key_schedule *);
int	DES_key_sched(DES_cblock *, DES_key_schedule *);
void	DES_string_to_key(const char *, DES_cblock *);
int	DES_read_password(DES_cblock *, char *, int);

void	HC_DEPRECATED DES_rand_data(void *, int);
void	HC_DEPRECATED DES_set_random_generator_seed(DES_cblock *);
void	HC_DEPRECATED DES_generate_random_block(DES_cblock *);
void	HC_DEPRECATED DES_set_sequence_number(void *);
void 	HC_DEPRECATED DES_init_random_number_generator(DES_cblock *);
void	HC_DEPRECATED DES_random_key(DES_cblock *);
int	HC_DEPRECATED DES_new_random_key(DES_cblock *);


void	DES_encrypt(uint32_t [2], DES_key_schedule *, int);
void	DES_ecb_encrypt(DES_cblock *, DES_cblock *, DES_key_schedule *, int);
void	DES_ecb3_encrypt(DES_cblock *,DES_cblock *, DES_key_schedule *,
			 DES_key_schedule *, DES_key_schedule *, int);
void	DES_pcbc_encrypt(const void *, void *, long,
			 DES_key_schedule *, DES_cblock *, int);
void	DES_cbc_encrypt(const void *, void *, long,
			DES_key_schedule *, DES_cblock *, int);
void	DES_ede3_cbc_encrypt(const void *, void *, long,
			     DES_key_schedule *, DES_key_schedule *,
			     DES_key_schedule *, DES_cblock *, int);
void	DES_cfb64_encrypt(const void *, void *, long,
			  DES_key_schedule *, DES_cblock *, int *, int);


uint32_t DES_cbc_cksum(const void *, DES_cblock *,
		      long, DES_key_schedule *, DES_cblock *);


void	_DES_ipfp_test(void);

#ifdef  __cplusplus
}
#endif


#endif /* _DESperate_H */
