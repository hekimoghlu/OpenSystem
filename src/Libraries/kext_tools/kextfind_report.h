/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 4, 2024.
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
#ifndef _KEXTFIND_REPORT_H_
#define _KEXTFIND_REPORT_H_

#include "QEQuery.h"
#include "kextfind_tables.h"
#include "kextfind_query.h"

/* These arenn't processed by getopt or QEQuery, we just look for them.
 */
#define kKeywordReport   "-report"
#define kNoReportHeader  "-no-header"

#define kPredNameSymbol  "-symbol"
#define kPredCharSymbol  "-sym"

/*******************************************************************************
* Query Engine Callbacks
*
* The Query Engine invokes these as it finds keywords from the above list
* in the command line or the query being reportEvaluated.
*******************************************************************************/
Boolean reportParseProperty(
    CFMutableDictionaryRef element,
    int argc,
    char * const argv[],
    uint32_t * num_used,
    void * user_data,
    QEQueryError * error);

Boolean reportParseShorthand(
    CFMutableDictionaryRef element,
    int argc,
    char * const argv[],
    uint32_t * num_used,
    void * user_data,
    QEQueryError * error);

Boolean reportEvalProperty(
    CFDictionaryRef element,
    void * object,
    void * user_data,
    QEQueryError * error);

Boolean reportEvalMatchProperty(
    CFDictionaryRef element,
    void * object,
    void * user_data,
    QEQueryError * error);

Boolean reportParseFlag(
    CFMutableDictionaryRef element,
    int argc,
    char * const argv[],
    uint32_t * num_used,
    void * user_data,
    QEQueryError * error);

Boolean reportEvalFlag(
    CFDictionaryRef element,
    void * object,
    void * user_data,
    QEQueryError * error);

Boolean reportParseArch(
    CFMutableDictionaryRef element,
    int argc,
    char * const argv[],
    uint32_t * num_used,
    void * user_data,
    QEQueryError * error);

Boolean reportEvalArch(
    CFDictionaryRef element,
    void * object,
    void * user_data,
    QEQueryError * error);

Boolean reportEvalArchExact(
    CFDictionaryRef element,
    void * object,
    void * user_data,
    QEQueryError * error);

Boolean reportParseCommand(
    CFMutableDictionaryRef element,
    int argc,
    char * const argv[],
    uint32_t * num_used,
    void * user_data,
    QEQueryError * error);

Boolean reportParseDefinesOrReferencesSymbol(
    CFMutableDictionaryRef element,
    int argc,
    char * const argv[],
    uint32_t * num_used,
    void * user_data,
    QEQueryError * error);

Boolean reportEvalDefinesOrReferencesSymbol(
    CFDictionaryRef element,
    void * object,
    void * user_data,
    QEQueryError * error);

Boolean reportEvalCommand(
    CFDictionaryRef element,
    void * object,
    void * user_data,
    QEQueryError * error);

#endif /* _KEXTFIND_REPORT_H_ */
