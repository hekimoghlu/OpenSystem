/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 19, 2024.
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

#ifndef __krb5cf_protos_h__
#define __krb5cf_protos_h__

#include <stdarg.h>
#include <CoreFoundation/CoreFoundation.h>

#include <krb5.h>
#include <krb5-protos.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Returns an array of dictionaries containing principal name
 * currently logged in, its audit session ID and its expiration time
 *
 * @return array of dictionaries containing principal name
 * currently logged in, its audit session ID and its expiration time
 * The array needs to be released.
 *
 * @ingroup krb5
 */

KRB5_LIB_FUNCTION CFArrayRef KRB5_LIB_CALL
krb5_kcm_get_principal_list(krb5_context context) CF_RETURNS_RETAINED;

#ifdef __cplusplus
}
#endif

#endif /* __krb5cf_protos_h__ */
