/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 31, 2023.
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
#ifndef _EAP8021X_DIAMETERAVP_H
#define _EAP8021X_DIAMETERAVP_H

/*
 * DiameterAVP.h
 * - definitions for Diameter AVP's
 */

/* 
 * Modification History
 *
 * October 11, 2002	Dieter Siegmund (dieter@apple)
 * - created
 */
typedef struct {
    u_int32_t	AVP_code;
    u_int32_t	AVP_flags_length;
    u_char	AVP_data[0];
} DiameterAVP;

typedef struct {
    u_int32_t	AVPV_code;
    u_int32_t	AVPV_flags_length;
    u_int32_t	AVPV_vendor;
    u_char	AVPV_data[0];
} DiameterVendorAVP;

typedef enum {
    kDiameterFlagsVendorSpecific = 0x80,
    kDiameterFlagsMandatory = 0x40,
} DiameterFlags;

#define DIAMETER_LENGTH_MASK	0xffffff

static __inline__ u_int32_t
DiameterAVPMakeFlagsLength(u_int8_t flags, u_int32_t length)
{
    u_int32_t flags_length;

    flags_length = (length & DIAMETER_LENGTH_MASK) | (flags << 24);
    return (flags_length);
}

static __inline__ u_int32_t
DiameterAVPLengthFromFlagsLength(u_int32_t flags_length)
{
    return (flags_length & DIAMETER_LENGTH_MASK);
}

static __inline__ u_int8_t
DiameterAVPFlagsFromFlagsLength(u_int32_t flags_length)
{
    return (flags_length >> 24);
}

#endif /* _EAP8021X_DIAMETERAVP_H */
