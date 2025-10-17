/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 5, 2025.
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
/*
 * $Id$
 */

#ifndef _HEIM_RAND_H
#define _HEIM_RAND_H 1

typedef struct RAND_METHOD RAND_METHOD;

#include <hcrypto/engine.h>

/* symbol renaming */
#define RAND_bytes hc_RAND_bytes
#define RAND_pseudo_bytes hc_RAND_pseudo_bytes
#define RAND_seed hc_RAND_seed
#define RAND_cleanup hc_RAND_cleanup
#define RAND_add hc_RAND_add
#define RAND_set_rand_method hc_RAND_set_rand_method
#define RAND_get_rand_method hc_RAND_get_rand_method
#define RAND_set_rand_engine hc_RAND_set_rand_engine
#define RAND_file_name hc_RAND_file_name
#define RAND_load_file hc_RAND_load_file
#define RAND_write_file hc_RAND_write_file
#define RAND_status hc_RAND_status
#define RAND_egd hc_RAND_egd
#define RAND_egd_bytes hc_RAND_egd_bytes
#define RAND_fortuna_method hc_RAND_fortuna_method
#define RAND_egd_method hc_RAND_egd_method
#define RAND_unix_method hc_RAND_unix_method
#define RAND_cc_method hc_RAND_cc_method
#define RAND_w32crypto_method hc_RAND_w32crypto_method

/*
 *
 */

struct RAND_METHOD
{
    void (*seed)(const void *, int);
    int (*bytes)(unsigned char *, int);
    void (*cleanup)(void);
    void (*add)(const void *, int, double);
    int (*pseudorand)(unsigned char *, int);
    int (*status)(void);
};

/*
 *
 */

int	RAND_bytes(void *, size_t num);
int	RAND_pseudo_bytes(void *, size_t);
void	RAND_seed(const void *, size_t);
void	RAND_cleanup(void);
void	RAND_add(const void *, size_t, double);

int	RAND_set_rand_method(const RAND_METHOD *);
const RAND_METHOD *
	RAND_get_rand_method(void);
int	RAND_set_rand_engine(ENGINE *);

const char *
	RAND_file_name(char *, size_t);
int	RAND_load_file(const char *, size_t);
int	RAND_write_file(const char *);
int	RAND_status(void);
int	RAND_egd(const char *);
int	RAND_egd_bytes(const char *, int);


const RAND_METHOD *	RAND_fortuna_method(void);
const RAND_METHOD *	RAND_unix_method(void);
const RAND_METHOD *	RAND_cc_method(void);
const RAND_METHOD *	RAND_egd_method(void);
const RAND_METHOD *	RAND_w32crypto_method(void);

#endif /* _HEIM_RAND_H */
