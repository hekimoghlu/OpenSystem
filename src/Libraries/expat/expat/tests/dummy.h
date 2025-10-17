/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 10, 2025.
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
#ifdef __cplusplus
extern "C" {
#endif

#ifndef XML_DUMMY_H
#  define XML_DUMMY_H

#  define DUMMY_START_DOCTYPE_HANDLER_FLAG (1UL << 0)
#  define DUMMY_END_DOCTYPE_HANDLER_FLAG (1UL << 1)
#  define DUMMY_ENTITY_DECL_HANDLER_FLAG (1UL << 2)
#  define DUMMY_NOTATION_DECL_HANDLER_FLAG (1UL << 3)
#  define DUMMY_ELEMENT_DECL_HANDLER_FLAG (1UL << 4)
#  define DUMMY_ATTLIST_DECL_HANDLER_FLAG (1UL << 5)
#  define DUMMY_COMMENT_HANDLER_FLAG (1UL << 6)
#  define DUMMY_PI_HANDLER_FLAG (1UL << 7)
#  define DUMMY_START_ELEMENT_HANDLER_FLAG (1UL << 8)
#  define DUMMY_START_CDATA_HANDLER_FLAG (1UL << 9)
#  define DUMMY_END_CDATA_HANDLER_FLAG (1UL << 10)
#  define DUMMY_UNPARSED_ENTITY_DECL_HANDLER_FLAG (1UL << 11)
#  define DUMMY_START_NS_DECL_HANDLER_FLAG (1UL << 12)
#  define DUMMY_END_NS_DECL_HANDLER_FLAG (1UL << 13)
#  define DUMMY_START_DOCTYPE_DECL_HANDLER_FLAG (1UL << 14)
#  define DUMMY_END_DOCTYPE_DECL_HANDLER_FLAG (1UL << 15)
#  define DUMMY_SKIP_HANDLER_FLAG (1UL << 16)
#  define DUMMY_DEFAULT_HANDLER_FLAG (1UL << 17)

extern void init_dummy_handlers(void);
extern unsigned long get_dummy_handler_flags(void);

extern void XMLCALL dummy_xdecl_handler(void *userData, const XML_Char *version,
                                        const XML_Char *encoding,
                                        int standalone);

extern void XMLCALL dummy_start_doctype_handler(void *userData,
                                                const XML_Char *doctypeName,
                                                const XML_Char *sysid,
                                                const XML_Char *pubid,
                                                int has_internal_subset);

extern void XMLCALL dummy_end_doctype_handler(void *userData);

extern void XMLCALL dummy_entity_decl_handler(
    void *userData, const XML_Char *entityName, int is_parameter_entity,
    const XML_Char *value, int value_length, const XML_Char *base,
    const XML_Char *systemId, const XML_Char *publicId,
    const XML_Char *notationName);

extern void XMLCALL dummy_notation_decl_handler(void *userData,
                                                const XML_Char *notationName,
                                                const XML_Char *base,
                                                const XML_Char *systemId,
                                                const XML_Char *publicId);

extern void XMLCALL dummy_element_decl_handler(void *userData,
                                               const XML_Char *name,
                                               XML_Content *model);

extern void XMLCALL dummy_attlist_decl_handler(
    void *userData, const XML_Char *elname, const XML_Char *attname,
    const XML_Char *att_type, const XML_Char *dflt, int isrequired);

extern void XMLCALL dummy_comment_handler(void *userData, const XML_Char *data);

extern void XMLCALL dummy_pi_handler(void *userData, const XML_Char *target,
                                     const XML_Char *data);

extern void XMLCALL dummy_start_element(void *userData, const XML_Char *name,
                                        const XML_Char **atts);

extern void XMLCALL dummy_end_element(void *userData, const XML_Char *name);

extern void XMLCALL dummy_start_cdata_handler(void *userData);

extern void XMLCALL dummy_end_cdata_handler(void *userData);

extern void XMLCALL dummy_cdata_handler(void *userData, const XML_Char *s,
                                        int len);

extern void XMLCALL dummy_start_namespace_decl_handler(void *userData,
                                                       const XML_Char *prefix,
                                                       const XML_Char *uri);

extern void XMLCALL dummy_end_namespace_decl_handler(void *userData,
                                                     const XML_Char *prefix);

extern void XMLCALL dummy_unparsed_entity_decl_handler(
    void *userData, const XML_Char *entityName, const XML_Char *base,
    const XML_Char *systemId, const XML_Char *publicId,
    const XML_Char *notationName);

extern void XMLCALL dummy_default_handler(void *userData, const XML_Char *s,
                                          int len);

extern void XMLCALL dummy_start_doctype_decl_handler(
    void *userData, const XML_Char *doctypeName, const XML_Char *sysid,
    const XML_Char *pubid, int has_internal_subset);

extern void XMLCALL dummy_end_doctype_decl_handler(void *userData);

extern void XMLCALL dummy_skip_handler(void *userData,
                                       const XML_Char *entityName,
                                       int is_parameter_entity);

#endif /* XML_DUMMY_H */

#ifdef __cplusplus
}
#endif
