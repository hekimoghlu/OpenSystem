/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 28, 2022.
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
/* $Id: bit.h,v 1.14 2007/06/19 23:47:16 tbox Exp $ */

#ifndef DNS_BIT_H
#define DNS_BIT_H 1

/*! \file dns/bit.h */

#include <isc/int.h>
#include <isc/boolean.h>

typedef isc_uint64_t dns_bitset_t;

#define DNS_BIT_SET(bit, bitset) \
     (*(bitset) |= ((dns_bitset_t)1 << (bit)))
#define DNS_BIT_CLEAR(bit, bitset) \
     (*(bitset) &= ~((dns_bitset_t)1 << (bit)))
#define DNS_BIT_CHECK(bit, bitset) \
     ISC_TF((*(bitset) & ((dns_bitset_t)1 << (bit))) \
	    == ((dns_bitset_t)1 << (bit)))

#endif /* DNS_BIT_H */

