/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 15, 2024.
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
#ifndef __EXSLT_H__
#define __EXSLT_H__

#include <libxml/tree.h>
#include <libxml/xpath.h>
#include "exsltexports.h"
#include "exsltconfig.h"

#ifdef __cplusplus
extern "C" {
#endif

EXSLTPUBVAR const char *exsltLibraryVersion;
EXSLTPUBVAR const int exsltLibexsltVersion;
EXSLTPUBVAR const int exsltLibxsltVersion;
EXSLTPUBVAR const int exsltLibxmlVersion;

/**
 * EXSLT_COMMON_NAMESPACE:
 *
 * Namespace for EXSLT common functions
 */
#define EXSLT_COMMON_NAMESPACE ((const xmlChar *) "http://exslt.org/common")
/**
 * EXSLT_CRYPTO_NAMESPACE:
 *
 * Namespace for EXSLT crypto functions
 */
#define EXSLT_CRYPTO_NAMESPACE ((const xmlChar *) "http://exslt.org/crypto")
/**
 * EXSLT_MATH_NAMESPACE:
 *
 * Namespace for EXSLT math functions
 */
#define EXSLT_MATH_NAMESPACE ((const xmlChar *) "http://exslt.org/math")
/**
 * EXSLT_SETS_NAMESPACE:
 *
 * Namespace for EXSLT set functions
 */
#define EXSLT_SETS_NAMESPACE ((const xmlChar *) "http://exslt.org/sets")
/**
 * EXSLT_FUNCTIONS_NAMESPACE:
 *
 * Namespace for EXSLT functions extension functions
 */
#define EXSLT_FUNCTIONS_NAMESPACE ((const xmlChar *) "http://exslt.org/functions")
/**
 * EXSLT_STRINGS_NAMESPACE:
 *
 * Namespace for EXSLT strings functions
 */
#define EXSLT_STRINGS_NAMESPACE ((const xmlChar *) "http://exslt.org/strings")
/**
 * EXSLT_DATE_NAMESPACE:
 *
 * Namespace for EXSLT date functions
 */
#define EXSLT_DATE_NAMESPACE ((const xmlChar *) "http://exslt.org/dates-and-times")
/**
 * EXSLT_DYNAMIC_NAMESPACE:
 *
 * Namespace for EXSLT dynamic functions
 */
#define EXSLT_DYNAMIC_NAMESPACE ((const xmlChar *) "http://exslt.org/dynamic")

/**
 * SAXON_NAMESPACE:
 *
 * Namespace for SAXON extensions functions
 */
#define SAXON_NAMESPACE ((const xmlChar *) "http://icl.com/saxon")

EXSLTPUBFUN void EXSLTCALL exsltCommonRegister (void);
#ifdef EXSLT_CRYPTO_ENABLED
EXSLTPUBFUN void EXSLTCALL exsltCryptoRegister (void);
#endif
EXSLTPUBFUN void EXSLTCALL exsltMathRegister (void);
EXSLTPUBFUN void EXSLTCALL exsltSetsRegister (void);
EXSLTPUBFUN void EXSLTCALL exsltFuncRegister (void);
EXSLTPUBFUN void EXSLTCALL exsltStrRegister (void);
EXSLTPUBFUN void EXSLTCALL exsltDateRegister (void);
EXSLTPUBFUN void EXSLTCALL exsltSaxonRegister (void);
EXSLTPUBFUN void EXSLTCALL exsltDynRegister(void);

EXSLTPUBFUN void EXSLTCALL exsltRegisterAll (void);

EXSLTPUBFUN int EXSLTCALL exsltDateXpathCtxtRegister (xmlXPathContextPtr ctxt,
                                                      const xmlChar *prefix);
EXSLTPUBFUN int EXSLTCALL exsltMathXpathCtxtRegister (xmlXPathContextPtr ctxt,
                                                      const xmlChar *prefix);
EXSLTPUBFUN int EXSLTCALL exsltSetsXpathCtxtRegister (xmlXPathContextPtr ctxt,
                                                      const xmlChar *prefix);
EXSLTPUBFUN int EXSLTCALL exsltStrXpathCtxtRegister (xmlXPathContextPtr ctxt,
                                                     const xmlChar *prefix);

#ifdef __cplusplus
}
#endif
#endif /* __EXSLT_H__ */

