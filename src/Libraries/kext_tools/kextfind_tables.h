/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 8, 2024.
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
#ifndef _KEXTFIND_TABLES_H_
#define _KEXTFIND_TABLES_H_

#include <CoreFoundation/CoreFoundation.h>
#include <getopt.h>
#include "QEQuery.h"
#include "kext_tools_util.h"

/*******************************************************************************
* This data structure associated query keywords with the parsing and evaluation
* function callbacks used by the query engine. Some callbacks handle several
* keywords because of similar arguments or evaluation logic.
*
* See kextfind_query.[hc] for the definitions of these things.
*******************************************************************************/
struct querySetup {
    CFStringRef longName;
    CFStringRef shortName;
    QEQueryParseCallback parseCallback;
    QEQueryEvaluationCallback evalCallback;
};

/*******************************************************************************
* Command-line options (as opposed to query predicate keywords).
* This data is used by getopt_long_only().
*******************************************************************************/
/* Would like a way to automatically combine these into the optstring.
 */
#define kOptNameHelp                    "help"
#define kOptNameCaseInsensitive         "case-insensitive"
#define kOptNameNulTerminate            "nul"
#define kOptNameSearchItem              "search-item"
#define kOptNameSubstring               "substring"
#define kOptNameDefaultArch             "set-arch"

#ifdef EXTRA_INFO
// I think there will be better ways to do this after getting some airtime
#define kOptNameExtraInfo               "extra-info"
#endif

#define kOptNameRelativePaths           "relative-paths"
#define kOptNameNoPaths                 "no-paths"

// Currently unused, although code does reference them
// Things are picky by default for now.
#define kOptNameMeek                    "meek"
#define kOptNamePicky                   "picky"

#define kOPT_CHARS  "0ef:his"

enum {
    kOptSystemExtensions = 'e',
    kOptCaseInsensitive = 'i',
    kOptNulTerminate = '0',
    kOptSearchItem = 'f',
    kOptSubstring = 's',
};

/* Options with no single-letter variant.  */
// Do not use -1, that's getopt() end-of-args return value
// and can cause confusion
enum {
    kLongOptQueryPredicate = -2,
#ifdef EXTRA_INFO
    kLongOptExtraInfo = -3,
#endif
    kLongOptRelativePaths = -4,
    kLongOptNoPaths = -5,
    kLongOptMeek = -6,
    kLongOptPicky = -7,
    kLongOptReport = -8,
    kLongOptDefaultArch = -9,
};

/*******************************************************************************
* The structure of info needed by getopt_long_only().
*******************************************************************************/
extern struct option opt_info[];
extern int           longopt;

/*******************************************************************************
* A list of predicate names, synonyms, and parse/eval callbacks for the
* query engine.
*******************************************************************************/
extern struct querySetup queryCallbackList[];
extern struct querySetup reportCallbackList[];


/*******************************************************************************
*
*******************************************************************************/
#define QUERY_PREDICATES  \
    { &kPredNameProperty[1],               no_argument, &longopt, kLongOptQueryPredicate },  \
    { &kPredNamePropertyExists[1],         no_argument, &longopt, kLongOptQueryPredicate },  \
    \
    { &kPredNameMatchProperty[1],          no_argument, &longopt, kLongOptQueryPredicate },  \
    { &kPredNameMatchPropertyExists[1],    no_argument, &longopt, kLongOptQueryPredicate },  \
    \
    { &kPredNameLoaded[1],                 no_argument, &longopt, kLongOptQueryPredicate },  \
    { &kPredNameValid[1],                  no_argument, &longopt, kLongOptQueryPredicate },  \
    { &kPredNameAuthentic[1],              no_argument, &longopt, kLongOptQueryPredicate },  \
    { &kPredNameDependenciesMet[1],        no_argument, &longopt, kLongOptQueryPredicate },  \
    { &kPredNameLoadable[1],               no_argument, &longopt, kLongOptQueryPredicate },  \
    { &kPredNameWarnings[1],               no_argument, &longopt, kLongOptQueryPredicate },  \
    { &kPredNameIsLibrary[1],              no_argument, &longopt, kLongOptQueryPredicate },  \
    \
    { &kQEQueryTokenAnd[1],                no_argument, &longopt, kLongOptQueryPredicate },  \
    { &kQEQueryTokenOr[1],                 no_argument, &longopt, kLongOptQueryPredicate },  \
    { &kQEQueryTokenNot[1],                no_argument, &longopt, kLongOptQueryPredicate },  \
    \
    { &kPredNameVersion[1],                no_argument, &longopt, kLongOptQueryPredicate },  \
    { &kPredNameCompatibleWithVersion[1],  no_argument, &longopt, kLongOptQueryPredicate },  \
    { &kPredNameIntegrity[1],              no_argument, &longopt, kLongOptQueryPredicate },  \
    \
    { &kPredNameHasPlugins[1],             no_argument, &longopt, kLongOptQueryPredicate },  \
    { &kPredNameIsPlugin[1],               no_argument, &longopt, kLongOptQueryPredicate },  \
    { &kPredNameHasDebugProperties[1],     no_argument, &longopt, kLongOptQueryPredicate },  \
    \
    { &kPredNameArch[1],                   no_argument, &longopt, kLongOptQueryPredicate },  \
    { &kPredNameArchExact[1],              no_argument, &longopt, kLongOptQueryPredicate },  \
    { &kPredNameExecutable[1],             no_argument, &longopt, kLongOptQueryPredicate },  \
    { &kPredNameNoExecutable[1],           no_argument, &longopt, kLongOptQueryPredicate },  \
    { &kPredNameDefinesSymbol[1],          no_argument, &longopt, kLongOptQueryPredicate },  \
    { &kPredNameReferencesSymbol[1],       no_argument, &longopt, kLongOptQueryPredicate },  \
    \
    { &kPredNameDuplicate[1],              no_argument, &longopt, kLongOptQueryPredicate },  \
    \
    { &kPredNameInvalid[1],                no_argument, &longopt, kLongOptQueryPredicate },  \
    { &kPredNameInauthentic[1],            no_argument, &longopt, kLongOptQueryPredicate },  \
    { &kPredNameDependenciesMissing[1],    no_argument, &longopt, kLongOptQueryPredicate },  \
    { &kPredNameNonloadable[1],            no_argument, &longopt, kLongOptQueryPredicate },  \
    \
    { &kPredNameBundleID[1],               no_argument, &longopt, kLongOptQueryPredicate },  \
    { &kPredNameBundleName[1],             no_argument, &longopt, kLongOptQueryPredicate },  \
    \
    { &kPredNameRoot[1],                   no_argument, &longopt, kLongOptQueryPredicate },  \
    { &kPredNameConsole[1],                no_argument, &longopt, kLongOptQueryPredicate },  \
    { &kPredNameLocalRoot[1],              no_argument, &longopt, kLongOptQueryPredicate },  \
    { &kPredNameNetworkRoot[1],            no_argument, &longopt, kLongOptQueryPredicate },  \
    { &kPredNameSafeBoot[1],               no_argument, &longopt, kLongOptQueryPredicate },  \
    \
    { &kPredCharProperty[1],               no_argument, &longopt, kLongOptQueryPredicate },  \
    { &kPredCharPropertyExists[1],         no_argument, &longopt, kLongOptQueryPredicate },  \
    \
    { &kPredCharMatchProperty[1],          no_argument, &longopt, kLongOptQueryPredicate },  \
    { &kPredCharMatchPropertyExists[1],    no_argument, &longopt, kLongOptQueryPredicate },  \
    \
    { &kPredCharValid[1],                  no_argument, &longopt, kLongOptQueryPredicate },  \
    { &kPredCharAuthentic[1],              no_argument, &longopt, kLongOptQueryPredicate },  \
    { &kPredCharDependenciesMet[1],        no_argument, &longopt, kLongOptQueryPredicate },  \
    { &kPredCharLoadable[1],               no_argument, &longopt, kLongOptQueryPredicate },  \
    { &kPredCharWarnings[1],               no_argument, &longopt, kLongOptQueryPredicate },  \
    \
    { &kQEQueryTokenAnd[1],                no_argument, &longopt, kLongOptQueryPredicate },  \
    { &kQEQueryTokenOr[1],                 no_argument, &longopt, kLongOptQueryPredicate },  \
    { &kQEQueryTokenNot[1],                no_argument, &longopt, kLongOptQueryPredicate },  \
    \
    { &kPredCharVersion[1],                no_argument, &longopt, kLongOptQueryPredicate },  \
    \
    { &kPredCharArchExact[1],              no_argument, &longopt, kLongOptQueryPredicate },  \
    { &kPredCharExecutable[1],             no_argument, &longopt, kLongOptQueryPredicate },  \
    { &kPredCharNoExecutable[1],           no_argument, &longopt, kLongOptQueryPredicate },  \
    { &kPredCharDefinesSymbol[1],          no_argument, &longopt, kLongOptQueryPredicate },  \
    { &kPredCharReferencesSymbol[1],       no_argument, &longopt, kLongOptQueryPredicate },  \
    \
    { &kPredCharDuplicate[1],              no_argument, &longopt, kLongOptQueryPredicate },  \
    \
    { &kPredCharInvalid[1],                no_argument, &longopt, kLongOptQueryPredicate },  \
    { &kPredCharInauthentic[1],            no_argument, &longopt, kLongOptQueryPredicate },  \
    { &kPredCharDependenciesMissing[1],    no_argument, &longopt, kLongOptQueryPredicate },  \
    { &kPredCharNonloadable[1],            no_argument, &longopt, kLongOptQueryPredicate },  \
    \
    { &kPredCharBundleID[1],               no_argument, &longopt, kLongOptQueryPredicate },  \
    \
    { &kPredCharRoot[1],                   no_argument, &longopt, kLongOptQueryPredicate },  \
    { &kPredCharConsole[1],                no_argument, &longopt, kLongOptQueryPredicate },  \
    { &kPredCharLocalRoot[1],              no_argument, &longopt, kLongOptQueryPredicate },  \
    { &kPredCharNetworkRoot[1],            no_argument, &longopt, kLongOptQueryPredicate },  \
    { &kPredCharSafeBoot[1],               no_argument, &longopt, kLongOptQueryPredicate },  \

  #define QUERY_COMMANDS  \
    { &kPredNameEcho[1],                   no_argument, &longopt, kLongOptQueryPredicate },  \
    { &kPredNamePrint[1],                  no_argument, &longopt, kLongOptQueryPredicate },  \
    { &kPredNamePrintDiagnostics[1],       no_argument, &longopt, kLongOptQueryPredicate },  \
    { &kPredNamePrintProperty[1],          no_argument, &longopt, kLongOptQueryPredicate },  \
    { &kPredNamePrintMatchProperty[1],     no_argument, &longopt, kLongOptQueryPredicate },  \
    { &kPredNamePrintArches[1],            no_argument, &longopt, kLongOptQueryPredicate },  \
    { &kPredNamePrintDependencies[1],      no_argument, &longopt, kLongOptQueryPredicate },  \
    { &kPredNamePrintDependents[1],        no_argument, &longopt, kLongOptQueryPredicate },  \
    { &kPredNamePrintIntegrity[1],         no_argument, &longopt, kLongOptQueryPredicate },  \
    { &kPredNamePrintPlugins[1],           no_argument, &longopt, kLongOptQueryPredicate },  \
    { &kPredNamePrintInfoDictionary[1],    no_argument, &longopt, kLongOptQueryPredicate },  \
    { &kPredNamePrintExecutable[1],        no_argument, &longopt, kLongOptQueryPredicate },  \
    \
    { &kPredNameExec[1],                   no_argument, &longopt, kLongOptQueryPredicate },  \
    { &kPredCharPrintDiagnostics[1],       no_argument, &longopt, kLongOptQueryPredicate },  \
    { &kPredCharPrintProperty[1],          no_argument, &longopt, kLongOptQueryPredicate },  \
    { &kPredCharPrintMatchProperty[1],     no_argument, &longopt, kLongOptQueryPredicate },  \
    { &kPredCharPrintArches[1],            no_argument, &longopt, kLongOptQueryPredicate },  \
    { &kPredCharPrintInfoDictionary[1],    no_argument, &longopt, kLongOptQueryPredicate },  \
    { &kPredCharPrintExecutable[1],        no_argument, &longopt, kLongOptQueryPredicate },


#endif /* _KEXTFIND_TABLES_H_ */
