/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 3, 2025.
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
#ifndef	_OCSPD_DEBUGGING_H_
#define _OCSPD_DEBUGGING_H_

#include <security_utilities/debugging.h>

#define ocspdErrorLog(args...)		secnotice("ocspdError", ## args)
#define ocspdDebug(args...)			secinfo("ocspd", ## args)
#define ocspdDbDebug(args...)		secinfo("ocspdDb", ## args)
#define ocspdCrlDebug(args...)		secinfo("ocspdCrlDebug", ## args)
#define ocspdTrustDebug(args...)	secinfo("ocspdTrustDebug", ## args)
#define ocspdHttpDebug(args...)	secinfo("ocspdHttp", ## args)
#define ocspdLdapDebug(args...)	secinfo("ocspdLdap", ## args)


#endif	/* _OCSPD_DEBUGGING_H_ */
