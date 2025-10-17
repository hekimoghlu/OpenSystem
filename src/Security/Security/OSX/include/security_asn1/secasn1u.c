/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 27, 2021.
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
 * Utility routines to complement the ASN.1 encoding and decoding functions.
 *
 * $Id: secasn1u.c,v 1.3 2004/05/13 15:29:13 dmitch Exp $
 */

#include "secasn1.h"


/*
 * We have a length that needs to be encoded; how many bytes will the
 * encoding take?
 *
 * The rules are that 0 - 0x7f takes one byte (the length itself is the
 * entire encoding); everything else takes one plus the number of bytes
 * in the length.
 */
unsigned long SEC_ASN1LengthLength(unsigned long len)
{
    unsigned long lenlen = 1;

    if (len > 0x7f) {
        do {
            lenlen++;
            len >>= 8;
        } while (len);
    }

    return lenlen;
}


/*
 * XXX Move over (and rewrite as appropriate) the rest of the
 * stuff in dersubr.c!
 */


/*
 * Find the appropriate subtemplate for the given template.
 * This may involve calling a "chooser" function, or it may just
 * be right there.  In either case, it is expected to *have* a
 * subtemplate; this is asserted in debug builds (in non-debug
 * builds, NULL will be returned).
 *
 * "thing" is a pointer to the structure being encoded/decoded
 * "encoding", when true, means that we are in the process of encoding
 *	(as opposed to in the process of decoding)
 */
const SecAsn1Template* SEC_ASN1GetSubtemplate(const SecAsn1Template* theTemplate,
                                              void* thing,
                                              PRBool encoding
#ifdef __APPLE__
                                              ,
                                              const char* buf,  // for decode only
                                              size_t len
#endif
)
{
    const SecAsn1Template* subt = NULL;

    PORT_Assert(theTemplate->sub != NULL);
    if (theTemplate->sub != NULL) {
        if (theTemplate->kind & SEC_ASN1_DYNAMIC) {
            SecAsn1TemplateChooserPtr chooserp;

            chooserp = *(SecAsn1TemplateChooserPtr*)theTemplate->sub;
            if (chooserp) {
                void* dest = thing;
                if (thing != NULL) {
                    thing = (char*)thing - theTemplate->offset;
                }
                subt = (*chooserp)(thing, encoding, buf, len, dest);
            }
        } else {
            subt = (SecAsn1Template*)theTemplate->sub;
        }
    }
    return subt;
}
