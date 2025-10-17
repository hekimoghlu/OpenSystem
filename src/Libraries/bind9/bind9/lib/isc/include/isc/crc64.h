/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 27, 2025.
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
#ifndef ISC_CRC64_H
#define ISC_CRC64_H 1

/*! \file isc/crc64.h
 * \brief CRC64 in C
 */

#include <isc/lang.h>
#include <isc/types.h>

ISC_LANG_BEGINDECLS

void
isc_crc64_init(isc_uint64_t *crc);
/*%
 * Initialize a new CRC.
 *
 * Requires:
 * * 'crc' is not NULL.
 */

void
isc_crc64_update(isc_uint64_t *crc, const void *data, size_t len);
/*%
 * Add data to the CRC.
 *
 * Requires:
 * * 'crc' is not NULL.
 * * 'data' is not NULL.
 */

void
isc_crc64_final(isc_uint64_t *crc);
/*%
 * Finalize the CRC.
 *
 * Requires:
 * * 'crc' is not NULL.
 */

ISC_LANG_ENDDECLS

#endif /* ISC_CRC64_H */
