/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 18, 2023.
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
 * pkcs12Debug.h
 */

#ifndef	_PKCS12_DEBUG_H_
#define _PKCS12_DEBUG_H_

#include <security_utilities/debugging.h>
#include <Security/cssmapple.h>

#ifdef	NDEBUG
/* this actually compiles to nothing */
#define p12ErrorLog(args...)		secinfo("p12Error", ## args)
#define p12LogCssmError(op, err)
#else
#define p12ErrorLog(args...)		printf(args)
#define p12LogCssmError(op, err)	cssmPerror(op, err)
#endif
#define p12EventLog(args...)		secnotice("p12Event", ## args)

/* individual debug loggers */
#define p12DecodeLog(args...)		secinfo("p12Decode", ## args)
#define p12EncodeLog(args...)		secinfo("p12Encode", ## args)
#define p12CryptoLog(args...)		secinfo("p12Crypto", ## args)

#endif	/* _PKCS12_TEMPLATES_H_ */

