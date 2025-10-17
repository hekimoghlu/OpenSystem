/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 19, 2024.
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
 *  ccdebug.h - CommonCrypto debug macros
 *
 */

#if defined (_WIN32)
#define __unused
#endif

#if defined(COMMON_DIGEST_FUNCTIONS) || defined(COMMON_CMAC_FUNCTIONS) || defined(COMMON_GCM_FUNCTIONS) \
    || defined (COMMON_AESSHOEFLY_FUNCTIONS) || defined(COMMON_CASTSHOEFLY_FUNCTIONS) \
    || defined (COMMON_CRYPTOR_FUNCTIONS) || defined(COMMON_HMAC_FUNCTIONS) \
    || defined(COMMON_KEYDERIVATION_FUNCTIONS) || defined(COMMON_SYMMETRIC_KEYWRAP_FUNCTIONS) \
    || defined(COMMON_RSA_FUNCTIONS) || defined(COMMON_EC_FUNCTIONS) || defined(COMMON_DH_FUNCTIONS) \
    || defined(COMMON_BIGNUM_FUNCTIONS) || defined(COMMON_RANDOM_FUNCTIONS)

    #define DIAGNOSTIC
#endif


#ifdef DIAGNOSTIC
    #include <os/log.h>
    #define CC_DEBUG_LOG(lvl,fmt, ...)      os_log(OS_LOG_TYPE_DEBUG, __FUNCTION__ ## "  " ## fmt , __VA_ARGS__)
#else
    #define CC_DEBUG_LOG(...)  
#endif /* DEBUG */


