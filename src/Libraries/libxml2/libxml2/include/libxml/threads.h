/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 17, 2025.
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
#ifndef __XML_THREADS_H__
#define __XML_THREADS_H__

#include <libxml/xmlversion.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * xmlMutex are a simple mutual exception locks.
 */
typedef struct _xmlMutex xmlMutex;
typedef xmlMutex *xmlMutexPtr;

/*
 * xmlRMutex are reentrant mutual exception locks.
 */
typedef struct _xmlRMutex xmlRMutex;
typedef xmlRMutex *xmlRMutexPtr;

#ifdef __cplusplus
}
#endif
#include <libxml/globals.h>
#ifdef __cplusplus
extern "C" {
#endif
XMLPUBFUN xmlMutexPtr XMLCALL
			xmlNewMutex	(void);
XMLPUBFUN void XMLCALL
			xmlMutexLock	(xmlMutexPtr tok);
XMLPUBFUN void XMLCALL
			xmlMutexUnlock	(xmlMutexPtr tok);
XMLPUBFUN void XMLCALL
			xmlFreeMutex	(xmlMutexPtr tok);

XMLPUBFUN xmlRMutexPtr XMLCALL
			xmlNewRMutex	(void);
XMLPUBFUN void XMLCALL
			xmlRMutexLock	(xmlRMutexPtr tok);
XMLPUBFUN void XMLCALL
			xmlRMutexUnlock	(xmlRMutexPtr tok);
XMLPUBFUN void XMLCALL
			xmlFreeRMutex	(xmlRMutexPtr tok);

/*
 * Library wide APIs.
 */
XMLPUBFUN void XMLCALL
			xmlInitThreads	(void);
XMLPUBFUN void XMLCALL
			xmlLockLibrary	(void);
XMLPUBFUN void XMLCALL
			xmlUnlockLibrary(void);
XMLPUBFUN int XMLCALL
			xmlGetThreadId	(void);
XMLPUBFUN int XMLCALL
			xmlIsMainThread	(void);
XMLPUBFUN void XMLCALL
			xmlCleanupThreads(void);
XMLPUBFUN xmlGlobalStatePtr XMLCALL
			xmlGetGlobalState(void);

#ifdef HAVE_PTHREAD_H
#elif defined(HAVE_WIN32_THREADS) && !defined(HAVE_COMPILER_TLS) && (!defined(LIBXML_STATIC) || defined(LIBXML_STATIC_FOR_DLL))
#if defined(LIBXML_STATIC_FOR_DLL)
int XMLCALL
xmlDllMain(void *hinstDLL, unsigned long fdwReason,
           void *lpvReserved);
#endif
#endif

#ifdef __cplusplus
}
#endif


#endif /* __XML_THREADS_H__ */
