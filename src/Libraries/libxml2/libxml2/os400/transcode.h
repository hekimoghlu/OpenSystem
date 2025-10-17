/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 8, 2023.
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
#ifndef _TRANSCODE_H_
#define _TRANSCODE_H_

#include <stdarg.h>
#include <libxml/dict.h>


XMLPUBFUN void          xmlZapDict(xmlDictPtr * dict);
XMLPUBFUN const char *  xmlTranscodeResult(const xmlChar * s,
        const char * encoding, xmlDictPtr * dict,
        void (*freeproc)(const void *));
XMLPUBFUN const xmlChar * xmlTranscodeString(const char * s,
        const char * encoding, xmlDictPtr * dict);
XMLPUBFUN const xmlChar * xmlTranscodeWString(const char * s,
        const char * encoding, xmlDictPtr * dict);
XMLPUBFUN const xmlChar * xmlTranscodeHString(const char * s,
        const char * encoding, xmlDictPtr * dict);

#ifndef XML_NO_SHORT_NAMES
/**
***     Since the above functions are generally called "inline" (i.e.: several
***             times nested in a single expression), define shorthand names
***             to minimize calling statement length.
**/

#define xmlTR   xmlTranscodeResult
#define xmlTS   xmlTranscodeString
#define xmlTW   xmlTranscodeWString
#define xmlTH   xmlTranscodeHstring
#endif

XMLPUBFUN const char *  xmlVasprintf(xmlDictPtr * dict, const char * encoding,
        const xmlChar * fmt, va_list args);

#endif
