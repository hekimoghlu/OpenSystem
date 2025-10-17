/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 19, 2025.
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
#ifndef _TPDEBUGGING_H_
#define _TPDEBUGGING_H_

#include <security_utilities/debugging.h>

/* If TP_USE_SYSLOG is defined and not 0, use syslog() for debug
 * logging in addition to invoking the secinfo macro (which, as of
 * 10.11, emits a os_log message of a syslog message.)
 */
#ifndef TP_USE_SYSLOG
#define TP_USE_SYSLOG	0
#endif

#if TP_USE_SYSLOG
#include <syslog.h>
#define tp_secinfo(scope, format...) \
{ \
	syslog(LOG_NOTICE, format); \
	secinfo(scope, format); \
}
#else
#define tp_secinfo(scope, format...) \
	secinfo(scope, format)
#endif

#ifdef	NDEBUG
/* this actually compiles to nothing */
#define tpErrorLog(args...)		tp_secinfo("tpError", ## args)
#else
#define tpErrorLog(args...)		printf(args)
#endif

#define tpDebug(args...)		tp_secinfo("tpDebug", ## args)
#define tpDbDebug(args...)		tp_secinfo("tpDbDebug", ## args)
#define tpCrlDebug(args...)		tp_secinfo("tpCrlDebug", ## args)
#define tpPolicyError(args...)	tp_secinfo("tpPolicy", ## args)
#define tpVfyDebug(args...)		tp_secinfo("tpVfyDebug", ## args)
#define tpAnchorDebug(args...)	tp_secinfo("tpAnchorDebug", ## args)
#define tpOcspDebug(args...)	tp_secinfo("tpOcsp", ## args)
#define tpOcspCacheDebug(args...)	tp_secinfo("tpOcspCache", ## args)
#define tpTrustSettingsDbg(args...)	tp_secinfo("tpTrustSettings", ## args)

#endif	/* _TPDEBUGGING_H_ */
