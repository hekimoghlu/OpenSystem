/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 30, 2024.
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
    File:       AvailabilityInternalPrivate.h
 
    Contains:   SPI_AVAILABLE macro implementation

*/
#ifndef __AVAILABILITY_INTERNAL_PRIVATE__
#define __AVAILABILITY_INTERNAL_PRIVATE__

#include <AvailabilityInternal.h>

/*
 * SPI Availability
 *
 * These macros complement their API counterparts, and behave the same
 * for Apple internal clients.
 *
 * All SPI macros will be transformed to API_UNAVAILABLE for the public SDK
 * to prevent 3rd party developers from using the symbol.
 *
 * SPI_AVAILABLE
 * <rdar://problem/37321035> API_PROHIBITED should support version numbers too
 * <rdar://problem/40864547> Define SPI_AVAILABLE as an alternative to API_PROHIBITED
 *
 * SPI_DEPRECATED
 * SPI_DEPRECATED_WITH_REPLACEMENT
 * <rdar://problem/41506001> For parity, define SPI_DEPRECATED to provide an SPI variant of API_DEPRECATED
 */ 

#if defined(__has_feature) && defined(__has_attribute)
 #if __has_attribute(availability)

// @@AVAILABILITY_MACRO_INTERFACE(__SPI_AVAILABLE,__API_AVAILABLE)@@
// @@AVAILABILITY_MACRO_INTERFACE(__SPI_AVAILABLE,__API_AVAILABLE_BEGIN,scoped_availablity=TRUE)@@

// @@AVAILABILITY_MACRO_INTERFACE(SPI_AVAILABLE,__API_AVAILABLE)@@
// @@AVAILABILITY_MACRO_INTERFACE(SPI_AVAILABLE,__API_AVAILABLE_BEGIN,scoped_availablity=TRUE)@@

// @@AVAILABILITY_MACRO_INTERFACE(__SPI_DEPRECATED,__API_DEPRECATED_MSG,argCount=1)@@
// @@AVAILABILITY_MACRO_INTERFACE(SPI_DEPRECATED,__API_DEPRECATED_MSG,argCount=1)@@

// @@AVAILABILITY_MACRO_INTERFACE(__SPI_DEPRECATED_WITH_REPLACEMENT,__API_DEPRECATED_REP,argCount=1)@@
// @@AVAILABILITY_MACRO_INTERFACE(SPI_DEPRECATED_WITH_REPLACEMENT,__API_DEPRECATED_REP,argCount=1)@@
   
 #endif /* __has_attribute(availability) */
#endif /*  #if defined(__has_feature) && defined(__has_attribute) */

#endif /* __AVAILABILITY_INTERNAL_PRIVATE__ */


