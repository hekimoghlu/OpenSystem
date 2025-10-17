/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 12, 2024.
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
 * nbo.h
 * - network byte order
 * - inlines to set/get values to/from network byte order
 */

#ifndef _S_NBO_H
#define _S_NBO_H

#include "symbol_scope.h"
#include <stdint.h>
#include <strings.h>
#include <sys/_endian.h>

/*
 * Function: net_uint16_set
 * Purpose:
 *   Set a field in a structure that's at least 16 bits to the given
 *   value, putting it into network byte order
 */
INLINE void
net_uint16_set(uint8_t * field, uint16_t value)
{
    uint16_t tmp_value = htons(value);
    bcopy((void *)&tmp_value, (void *)field,
	  sizeof(uint16_t));
    return;
}

/*
 * Function: net_uint16_get
 * Purpose:
 *   Get a field in a structure that's at least 16 bits, converting
 *   to host byte order.
 */
INLINE uint16_t
net_uint16_get(const uint8_t * field)
{
    uint16_t tmp_field;

    bcopy((void *)field, (void *)&tmp_field, 
	  sizeof(uint16_t));
    return (ntohs(tmp_field));
}

/*
 * Function: net_uint32_set
 * Purpose:
 *   Set a field in a structure that's at least 32 bits to the given
 *   value, putting it into network byte order
 */
INLINE void
net_uint32_set(uint8_t * field, uint32_t value)
{
    uint32_t tmp_value = htonl(value);
    
    bcopy((void *)&tmp_value, (void *)field, 
	  sizeof(uint32_t));
    return;
}

/*
 * Function: net_uint32_get
 * Purpose:
 *   Get a field in a structure that's at least 32 bits, converting
 *   to host byte order.
 */
INLINE uint32_t
net_uint32_get(const uint8_t * field)
{
    uint32_t tmp_field;

    bcopy((void *)field, &tmp_field, 
	  sizeof(uint32_t));
    return (ntohl(tmp_field));
}

#endif /* _S_NBO_H */
