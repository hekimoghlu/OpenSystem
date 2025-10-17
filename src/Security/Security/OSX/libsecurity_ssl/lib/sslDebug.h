/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 30, 2024.
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
 * sslDebug.h - Debugging macros.
 */

#ifndef	_SSL_DEBUG_H_
#define _SSL_DEBUG_H_

#ifdef KERNEL
/* TODO: support secinfo in the kernel */
#define secinfo(x...)
#else /* KERNEL */
#include <utilities/debugging.h>
#endif

#ifndef	NDEBUG
#include <AssertMacros.h>
#endif


#define ssl_secinfo secinfo

#ifndef NDEBUG

/* log changes in handshake state */
#define sslHdskStateDebug(args...)		ssl_secinfo("sslHdskState", ## args)

/* log handshake and alert messages */
#define sslHdskMsgDebug(args...)		ssl_secinfo("sslHdskMsg", ## args)

/* log negotiated handshake parameters */
#define sslLogNegotiateDebug(args...)	ssl_secinfo("sslLogNegotiate", ## args)

/* log received protocol messsages */
#define sslLogRxProtocolDebug(msgType)	ssl_secinfo("sslLogRxProtocol", \
										"---received protoMsg %s", msgType)

/* log resumable session info */
#define sslLogResumSessDebug(args...)	ssl_secinfo("sslResumSession", ## args)

/* log low-level session info in appleSession.c */
#define sslLogSessCacheDebug(args...)	ssl_secinfo("sslSessionCache", ## args)

/* log record-level I/O (SSLRead, SSLWrite) */
#define sslLogRecordIo(args...)			ssl_secinfo("sslRecordIo", ## args)

/* cert-related info */
#define sslCertDebug(args...)			ssl_secinfo("sslCert", ## args)

/* Diffie-Hellman */
#define sslDhDebug(args...)				ssl_secinfo("sslDh", ## args)

/* EAP-FAST PAC-based session resumption */
#define sslEapDebug(args...)			ssl_secinfo("sslEap", ## args)

/* ECDSA */
#define sslEcdsaDebug(args...)			ssl_secinfo("sslEcdsa", ## args)

#else /* NDEBUG */

/*  deployment build */
#define sslHdskStateDebug(args...)
#define sslHdskMsgDebug(args...)
#define sslLogNegotiateDebug(args...)
#define sslLogRxProtocolDebug(msgType)
#define sslLogResumSessDebug(args...)
#define sslLogSessCacheDebug(args...)
#define sslLogRecordIo(args...)
#define sslCertDebug(args...)
#define sslDhDebug(args...)
#define sslEapDebug(args...)
#define sslEcdsaDebug(args...)

#endif	/*
NDEBUG */

#ifdef	NDEBUG

/* all errors logged to stdout for DEBUG config only */
#define sslErrorLog(args...)
#define sslDebugLog(args...)
#define sslDump(d, l)

#else

extern void SSLDump(const unsigned char *data, unsigned long len);

/* extra debug logging of non-error conditions, if SSL_DEBUG is defined */
#if SSL_DEBUG
//#define sslDebugLog(args...)        printf(args)
#define sslDebugLog(args...)        ssl_secinfo("sslDebug", ## args)
#else
#define sslDebugLog(args...)
#endif
/* all errors logged to stdout for DEBUG config only */
//#define sslErrorLog(args...)        printf(args)
#define sslErrorLog(args...)        ssl_secinfo("sslError", ## args)
#define sslDump(d, l)               SSLDump((d), (l))

#endif	/* NDEBUG */


#ifdef	NDEBUG
#define ASSERT(s)
#else
#define ASSERT(s)	check(s)
#endif

#endif	/* _SSL_DEBUG_H_ */
