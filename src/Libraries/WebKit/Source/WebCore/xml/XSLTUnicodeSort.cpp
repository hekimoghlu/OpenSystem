/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 24, 2021.
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
#include "config.h"
#include "XSLTUnicodeSort.h"

#if ENABLE(XSLT)

#include <algorithm>
#include <array>
#include <libxslt/templates.h>
#include <libxslt/xsltutils.h>
#include <wtf/Vector.h>
#include <wtf/unicode/Collator.h>

namespace WebCore {

// Based on default implementation from libxslt 1.1.22 and xsltICUSort.c example.
void xsltUnicodeSortFunction(xsltTransformContextPtr ctxt, xmlNodePtr* rawSorts, int nbsorts)
{
    if (!ctxt || !rawSorts || nbsorts <= 0 || nbsorts >= XSLT_MAX_SORT)
        return;

    auto sorts = unsafeMakeSpan(rawSorts, nbsorts);
    if (!sorts[0])
        return;

    auto comp = static_cast<xsltStylePreComp*>(sorts[0]->psvi);
    if (!comp)
        return;

    auto list = ctxt->nodeList;
    if (!list || list->nodeNr <= 1)
        return; /* nothing to do */

    std::array<int, XSLT_MAX_SORT> desc;
    std::array<int, XSLT_MAX_SORT> number;
    for (size_t j = 0; j < sorts.size(); ++j) {
        comp = static_cast<xsltStylePreComp*>(sorts[j]->psvi);
        if (!comp->stype && comp->has_stype) {
            auto* stype = xsltEvalAttrValueTemplate(ctxt, sorts[j], reinterpret_cast<const xmlChar*>("data-type"), XSLT_NAMESPACE);
            number[j] = 0;
            if (stype) {
                if (xmlStrEqual(stype, reinterpret_cast<const xmlChar*>("text"))) {
                    // number[j] already zero.
                } else if (xmlStrEqual(stype, reinterpret_cast<const xmlChar*>("number")))
                    number[j] = 1;
                else
                    xsltTransformError(ctxt, nullptr, sorts[j], "xsltDoSortFunction: no support for data-type = %s\n", stype);
// FIXME (286277): Stop ignoring -Wundef and -Wdeprecated-declarations in code that imports libxml and libxslt headers
IGNORE_WARNINGS_BEGIN("deprecated-declarations")
                xmlFree(stype);
IGNORE_WARNINGS_END
            }
        } else
            number[j] = comp->number;
        if (!comp->order && comp->has_order) {
            auto* order = xsltEvalAttrValueTemplate(ctxt, sorts[j], reinterpret_cast<const xmlChar*>("order"), XSLT_NAMESPACE);
            desc[j] = 0;
            if (order) {
                if (xmlStrEqual(order, reinterpret_cast<const xmlChar*>("ascending"))) {
                    // desc[j] already zero.
                } else if (xmlStrEqual(order, reinterpret_cast<const xmlChar*>("descending")))
                    desc[j] = 1;
                else
                    xsltTransformError(ctxt, nullptr, sorts[j], "xsltDoSortFunction: invalid value %s for order\n", order);
// FIXME (286277): Stop ignoring -Wundef and -Wdeprecated-declarations in code that imports libxml and libxslt headers
IGNORE_WARNINGS_BEGIN("deprecated-declarations")
                xmlFree(order);
IGNORE_WARNINGS_END
            }
        } else
            desc[j] = comp->descending;
    }

    auto len = list->nodeNr;
    auto listNodes = unsafeMakeSpan(list->nodeTab, len);

    std::array<std::span<xmlXPathObjectPtr>, XSLT_MAX_SORT> resultsTab = { };
    resultsTab[0] = unsafeMakeSpan(xsltComputeSortResult(ctxt, sorts[0]), len);

    auto results = resultsTab[0];
    if (!results.data())
        return;

    comp = static_cast<xsltStylePreComp*>(sorts[0]->psvi);

    // We are passing a language identifier to a function that expects a locale identifier.
    // The implementation of Collator should be lenient, and accept both "en-US" and "en_US", for example.
    // This lets an author specify sorting rules, e.g. "de_DE@collation=phonebook", which isn't
    // possible with language alone.
    Collator collator(comp->has_lang ? byteCast<char>(comp->lang) : "en", comp->lower_first);

    int depth = 0;
    int tst = 0;
    std::span<xmlXPathObjectPtr> res;
    /* Shell's sort of node-set */
    for (int incr = len / 2; incr > 0; incr /= 2) {
        for (int i = incr; i < len; i++) {
            int j = i - incr;
            if (!results[i])
                continue;
            
            while (j >= 0) {
                if (!results[j])
                    tst = 1;
                else {
                    if (number[0]) {
                        /* We make NaN smaller than number in accordance
                           with XSLT spec */
                        if (xmlXPathIsNaN(results[j]->floatval)) {
                            if (xmlXPathIsNaN(results[j + incr]->floatval))
                                tst = 0;
                            else
                                tst = -1;
                        } else if (xmlXPathIsNaN(results[j + incr]->floatval))
                            tst = 1;
                        else if (results[j]->floatval ==
                                results[j + incr]->floatval)
                            tst = 0;
                        else if (results[j]->floatval > 
                                results[j + incr]->floatval)
                            tst = 1;
                        else tst = -1;
                    } else
                        tst = collator.collate(byteCast<char8_t>(results[j]->stringval), byteCast<char8_t>(results[j + incr]->stringval));
                    if (desc[0])
                        tst = -tst;
                }
                if (tst == 0) {
                    /*
                     * Okay we need to use multi level sorts
                     */
                    depth = 1;
                    while (depth < nbsorts) {
                        if (!sorts[depth])
                            break;
                        comp = static_cast<xsltStylePreComp*>(sorts[depth]->psvi);
                        if (!comp)
                            break;

                        /*
                         * Compute the result of the next level for the
                         * full set, this might be optimized ... or not
                         */
                        if (!resultsTab[depth].data())
                            resultsTab[depth] = unsafeMakeSpan(xsltComputeSortResult(ctxt, sorts[depth]), len);
                        res = resultsTab[depth];
                        if (!res.data())
                            break;
                        if (!res[j]) {
                            if (res[j + incr])
                                tst = 1;
                        } else {
                            if (number[depth]) {
                                /* We make NaN smaller than number in
                                   accordance with XSLT spec */
                                if (xmlXPathIsNaN(res[j]->floatval)) {
                                    if (xmlXPathIsNaN(res[j +
                                                    incr]->floatval))
                                        tst = 0;
                                    else
                                        tst = -1;
                                } else if (xmlXPathIsNaN(res[j + incr]->
                                                floatval))
                                    tst = 1;
                                else if (res[j]->floatval == res[j + incr]->
                                                floatval)
                                    tst = 0;
                                else if (res[j]->floatval > 
                                        res[j + incr]->floatval)
                                    tst = 1;
                                else tst = -1;
                            } else
                                tst = collator.collate(byteCast<char8_t>(res[j]->stringval), byteCast<char8_t>(res[j + incr]->stringval));
                            if (desc[depth])
                                tst = -tst;
                        }

                        /*
                         * if we still can't differenciate at this level
                         * try one level deeper.
                         */
                        if (tst != 0)
                            break;
                        depth++;
                    }
                }
                if (tst == 0) {
                    tst = results[j]->index > results[j + incr]->index;
                }
                if (tst > 0) {
                    auto tmp = results[j];
                    results[j] = results[j + incr];
                    results[j + incr] = tmp;
                    auto node = listNodes[j];
                    listNodes[j] = listNodes[j + incr];
                    listNodes[j + incr] = node;
                    depth = 1;
                    while (depth < nbsorts) {
                        if (!sorts[depth])
                            break;
                        if (!resultsTab[depth].data())
                            break;
                        res = resultsTab[depth];
                        tmp = res[j];
                        res[j] = res[j + incr];
                        res[j + incr] = tmp;
                        depth++;
                    }
                    j -= incr;
                } else
                    break;
            }
        }
    }

    for (size_t j = 0; j < sorts.size(); ++j) {
        if (resultsTab[j].data()) {
            for (int i = 0; i < len; ++i)
                xmlXPathFreeObject(resultsTab[j][i]);
// FIXME (286277): Stop ignoring -Wundef and -Wdeprecated-declarations in code that imports libxml and libxslt headers
IGNORE_WARNINGS_BEGIN("deprecated-declarations")
            xmlFree(resultsTab[j].data());
IGNORE_WARNINGS_END
        }
    }
}

}

#endif
