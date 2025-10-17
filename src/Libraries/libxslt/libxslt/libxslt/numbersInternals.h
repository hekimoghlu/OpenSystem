/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 7, 2025.
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
#ifndef __XML_XSLT_NUMBERSINTERNALS_H__
#define __XML_XSLT_NUMBERSINTERNALS_H__

#include <libxml/tree.h>
#include "xsltexports.h"

#ifdef __cplusplus
extern "C" {
#endif

struct _xsltCompMatch;

/**
 * xsltNumberData:
 *
 * This data structure is just a wrapper to pass xsl:number data in.
 */
typedef struct _xsltNumberData xsltNumberData;
typedef xsltNumberData *xsltNumberDataPtr;

struct _xsltNumberData {
    const xmlChar *level;
    const xmlChar *count;
    const xmlChar *from;
    const xmlChar *value;
    const xmlChar *format;
    int has_format;
    int digitsPerGroup;
    int groupingCharacter;
    int groupingCharacterLen;
    xmlDocPtr doc;
    xmlNodePtr node;
    struct _xsltCompMatch *countPat;
    struct _xsltCompMatch *fromPat;

    /*
     * accelerators
     */
};

/**
 * xsltFormatNumberInfo,:
 *
 * This data structure lists the various parameters needed to format numbers.
 */
typedef struct _xsltFormatNumberInfo xsltFormatNumberInfo;
typedef xsltFormatNumberInfo *xsltFormatNumberInfoPtr;

struct _xsltFormatNumberInfo {
    int	    integer_hash;	/* Number of '#' in integer part */
    int	    integer_digits;	/* Number of '0' in integer part */
    int	    frac_digits;	/* Number of '0' in fractional part */
    int	    frac_hash;		/* Number of '#' in fractional part */
    int	    group;		/* Number of chars per display 'group' */
    int     multiplier;		/* Scaling for percent or permille */
    char    add_decimal;	/* Flag for whether decimal point appears in pattern */
    char    is_multiplier_set;	/* Flag to catch multiple occurences of percent/permille */
    char    is_negative_pattern;/* Flag for processing -ve prefix/suffix */
};

#ifdef __cplusplus
}
#endif
#endif /* __XML_XSLT_NUMBERSINTERNALS_H__ */
