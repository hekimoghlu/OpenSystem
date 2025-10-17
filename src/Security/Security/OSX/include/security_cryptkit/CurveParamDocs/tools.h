/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 10, 2023.
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
#define NUM_PRIMES 6542 /* PrimePi[2^16]. */
#define MILLER_RABIN_DEPTH (8)

void
init_tools(int shorts);

void
make_primes();

int
prime_literal(
	unsigned int	p
);

int
primeq(
	unsigned int 	odd
);

void
make_primes();

int
prime_probable(giant p);

int
jacobi_symbol(giant a, giant n);

int
pseudoq(giant a, giant p);

int
pseudointq(int a, giant p);


void
powFp2(giant a, giant b, giant w2, giant n, giant p);

int
sqrtmod(giant p, giant x);

void
sqrtg(giant n);

int
cornacchia4(giant n, int d, giant u, giant v);


