/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 31, 2022.
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
#ifndef __XML_EXSLTCONFIG_H__
#define __XML_EXSLTCONFIG_H__

#ifdef __cplusplus
extern "C" {
#endif

/**
 * LIBEXSLT_DOTTED_VERSION:
 *
 * the version string like "1.2.3"
 */
#define LIBEXSLT_DOTTED_VERSION "0.8.20"

/**
 * LIBEXSLT_VERSION:
 *
 * the version number: 1.2.3 value is 10203
 */
#define LIBEXSLT_VERSION 820

/**
 * LIBEXSLT_VERSION_STRING:
 *
 * the version number string, 1.2.3 value is "10203"
 */
#define LIBEXSLT_VERSION_STRING "820"

/**
 * LIBEXSLT_VERSION_EXTRA:
 *
 * extra version information, used to show a CVS compilation
 */
#define	LIBEXSLT_VERSION_EXTRA ""

/**
 * WITH_CRYPTO:
 *
 * Whether crypto support is configured into exslt
 */
#if 0
#define EXSLT_CRYPTO_ENABLED
#endif

/**
 * ATTRIBUTE_UNUSED:
 *
 * This macro is used to flag unused function parameters to GCC
 */
#ifdef __GNUC__
#ifndef ATTRIBUTE_UNUSED
#define ATTRIBUTE_UNUSED __attribute__((unused))
#endif
#else
#define ATTRIBUTE_UNUSED
#endif

#ifdef __cplusplus
}
#endif

#endif /* __XML_EXSLTCONFIG_H__ */
