/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 31, 2024.
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

#ifndef XML_HANDLERS_H
#  define XML_HANDLERS_H

#  include "expat_config.h"

#  include "expat.h"

/* Variable holding the expected handler userData */
extern const void *g_handler_data;
/* Count of the number of times the comment handler has been invoked */
extern int g_comment_count;
/* Count of the number of skipped entities */
extern int g_skip_count;
/* Count of the number of times the XML declaration handler is invoked */
extern int g_xdecl_count;

/* Start/End Element Handlers */

extern void XMLCALL start_element_event_handler(void *userData,
                                                const XML_Char *name,
                                                const XML_Char **atts);

extern void XMLCALL end_element_event_handler(void *userData,
                                              const XML_Char *name);

#  define STRUCT_START_TAG 0
#  define STRUCT_END_TAG 1

extern void XMLCALL start_element_event_handler2(void *userData,
                                                 const XML_Char *name,
                                                 const XML_Char **attr);

extern void XMLCALL end_element_event_handler2(void *userData,
                                               const XML_Char *name);

typedef struct attrInfo {
  const XML_Char *name;
  const XML_Char *value;
} AttrInfo;

typedef struct elementInfo {
  const XML_Char *name;
  int attr_count;
  const XML_Char *id_name;
  AttrInfo *attributes;
} ElementInfo;

typedef struct StructParserAndElementInfo {
  XML_Parser parser;
  ElementInfo *info;
} ParserAndElementInfo;

extern void XMLCALL counting_start_element_handler(void *userData,
                                                   const XML_Char *name,
                                                   const XML_Char **atts);

extern void XMLCALL suspending_end_handler(void *userData, const XML_Char *s);

extern void XMLCALL start_element_suspender(void *userData,
                                            const XML_Char *name,
                                            const XML_Char **atts);

extern int g_triplet_start_flag;
extern int g_triplet_end_flag;

extern void XMLCALL triplet_start_checker(void *userData, const XML_Char *name,
                                          const XML_Char **atts);

extern void XMLCALL triplet_end_checker(void *userData, const XML_Char *name);

extern void XMLCALL overwrite_start_checker(void *userData,
                                            const XML_Char *name,
                                            const XML_Char **atts);

extern void XMLCALL overwrite_end_checker(void *userData, const XML_Char *name);

extern void XMLCALL start_element_fail(void *userData, const XML_Char *name,
                                       const XML_Char **atts);

extern void XMLCALL start_ns_clearing_start_element(void *userData,
                                                    const XML_Char *prefix,
                                                    const XML_Char *uri);

typedef struct {
  XML_Parser parser;
  int deep;
} DataIssue240;

extern void XMLCALL start_element_issue_240(void *userData,
                                            const XML_Char *name,
                                            const XML_Char **atts);

extern void XMLCALL end_element_issue_240(void *userData, const XML_Char *name);

/* Text encoding handlers */

extern int XMLCALL UnknownEncodingHandler(void *data, const XML_Char *encoding,
                                          XML_Encoding *info);

extern int XMLCALL UnrecognisedEncodingHandler(void *data,
                                               const XML_Char *encoding,
                                               XML_Encoding *info);

extern int XMLCALL unknown_released_encoding_handler(void *data,
                                                     const XML_Char *encoding,
                                                     XML_Encoding *info);

extern int XMLCALL MiscEncodingHandler(void *data, const XML_Char *encoding,
                                       XML_Encoding *info);

extern int XMLCALL long_encoding_handler(void *userData,
                                         const XML_Char *encoding,
                                         XML_Encoding *info);

/* External Entity Handlers */

typedef struct ExtOption {
  const XML_Char *system_id;
  const char *parse_text;
} ExtOption;

extern int XMLCALL external_entity_optioner(XML_Parser parser,
                                            const XML_Char *context,
                                            const XML_Char *base,
                                            const XML_Char *systemId,
                                            const XML_Char *publicId);

extern int XMLCALL external_entity_loader(XML_Parser parser,
                                          const XML_Char *context,
                                          const XML_Char *base,
                                          const XML_Char *systemId,
                                          const XML_Char *publicId);

typedef struct ext_faults {
  const char *parse_text;
  const char *fail_text;
  const XML_Char *encoding;
  enum XML_Error error;
} ExtFaults;

extern int XMLCALL external_entity_faulter(XML_Parser parser,
                                           const XML_Char *context,
                                           const XML_Char *base,
                                           const XML_Char *systemId,
                                           const XML_Char *publicId);
extern int XMLCALL external_entity_failer__if_not_xml_ge(
    XML_Parser parser, const XML_Char *context, const XML_Char *base,
    const XML_Char *systemId, const XML_Char *publicId);
extern int XMLCALL external_entity_null_loader(XML_Parser parser,
                                               const XML_Char *context,
                                               const XML_Char *base,
                                               const XML_Char *systemId,
                                               const XML_Char *publicId);

extern int XMLCALL external_entity_resetter(XML_Parser parser,
                                            const XML_Char *context,
                                            const XML_Char *base,
                                            const XML_Char *systemId,
                                            const XML_Char *publicId);

extern int XMLCALL external_entity_suspender(XML_Parser parser,
                                             const XML_Char *context,
                                             const XML_Char *base,
                                             const XML_Char *systemId,
                                             const XML_Char *publicId);

extern int XMLCALL external_entity_suspend_xmldecl(XML_Parser parser,
                                                   const XML_Char *context,
                                                   const XML_Char *base,
                                                   const XML_Char *systemId,
                                                   const XML_Char *publicId);

extern int XMLCALL external_entity_suspending_faulter(XML_Parser parser,
                                                      const XML_Char *context,
                                                      const XML_Char *base,
                                                      const XML_Char *systemId,
                                                      const XML_Char *publicId);

extern int XMLCALL external_entity_cr_catcher(XML_Parser parser,
                                              const XML_Char *context,
                                              const XML_Char *base,
                                              const XML_Char *systemId,
                                              const XML_Char *publicId);

extern int XMLCALL external_entity_bad_cr_catcher(XML_Parser parser,
                                                  const XML_Char *context,
                                                  const XML_Char *base,
                                                  const XML_Char *systemId,
                                                  const XML_Char *publicId);

extern int XMLCALL external_entity_rsqb_catcher(XML_Parser parser,
                                                const XML_Char *context,
                                                const XML_Char *base,
                                                const XML_Char *systemId,
                                                const XML_Char *publicId);

extern int XMLCALL external_entity_good_cdata_ascii(XML_Parser parser,
                                                    const XML_Char *context,
                                                    const XML_Char *base,
                                                    const XML_Char *systemId,
                                                    const XML_Char *publicId);

/* Entity declaration handlers */

extern void XMLCALL entity_suspending_decl_handler(void *userData,
                                                   const XML_Char *name,
                                                   XML_Content *model);

extern void XMLCALL entity_suspending_xdecl_handler(void *userData,
                                                    const XML_Char *version,
                                                    const XML_Char *encoding,
                                                    int standalone);

extern int XMLCALL external_entity_param_checker(XML_Parser parser,
                                                 const XML_Char *context,
                                                 const XML_Char *base,
                                                 const XML_Char *systemId,
                                                 const XML_Char *publicId);

extern int XMLCALL external_entity_ref_param_checker(XML_Parser parameter,
                                                     const XML_Char *context,
                                                     const XML_Char *base,
                                                     const XML_Char *systemId,
                                                     const XML_Char *publicId);

extern int XMLCALL external_entity_param(XML_Parser parser,
                                         const XML_Char *context,
                                         const XML_Char *base,
                                         const XML_Char *systemId,
                                         const XML_Char *publicId);

extern int XMLCALL external_entity_load_ignore(XML_Parser parser,
                                               const XML_Char *context,
                                               const XML_Char *base,
                                               const XML_Char *systemId,
                                               const XML_Char *publicId);

extern int XMLCALL external_entity_load_ignore_utf16(XML_Parser parser,
                                                     const XML_Char *context,
                                                     const XML_Char *base,
                                                     const XML_Char *systemId,
                                                     const XML_Char *publicId);

extern int XMLCALL external_entity_load_ignore_utf16_be(
    XML_Parser parser, const XML_Char *context, const XML_Char *base,
    const XML_Char *systemId, const XML_Char *publicId);

extern int XMLCALL external_entity_valuer(XML_Parser parser,
                                          const XML_Char *context,
                                          const XML_Char *base,
                                          const XML_Char *systemId,
                                          const XML_Char *publicId);

extern int XMLCALL external_entity_not_standalone(XML_Parser parser,
                                                  const XML_Char *context,
                                                  const XML_Char *base,
                                                  const XML_Char *systemId,
                                                  const XML_Char *publicId);

extern int XMLCALL external_entity_value_aborter(XML_Parser parser,
                                                 const XML_Char *context,
                                                 const XML_Char *base,
                                                 const XML_Char *systemId,
                                                 const XML_Char *publicId);

extern int XMLCALL external_entity_public(XML_Parser parser,
                                          const XML_Char *context,
                                          const XML_Char *base,
                                          const XML_Char *systemId,
                                          const XML_Char *publicId);

extern int XMLCALL external_entity_devaluer(XML_Parser parser,
                                            const XML_Char *context,
                                            const XML_Char *base,
                                            const XML_Char *systemId,
                                            const XML_Char *publicId);

typedef struct ext_hdlr_data {
  const char *parse_text;
  XML_ExternalEntityRefHandler handler;
  CharData *storage;
} ExtHdlrData;

extern int XMLCALL external_entity_oneshot_loader(XML_Parser parser,
                                                  const XML_Char *context,
                                                  const XML_Char *base,
                                                  const XML_Char *systemId,
                                                  const XML_Char *publicId);

typedef struct ExtTest2 {
  const char *parse_text;
  int parse_len;
  const XML_Char *encoding;
  CharData *storage;
} ExtTest2;

extern int XMLCALL external_entity_loader2(XML_Parser parser,
                                           const XML_Char *context,
                                           const XML_Char *base,
                                           const XML_Char *systemId,
                                           const XML_Char *publicId);

typedef struct ExtFaults2 {
  const char *parse_text;
  int parse_len;
  const char *fail_text;
  const XML_Char *encoding;
  enum XML_Error error;
} ExtFaults2;

extern int XMLCALL external_entity_faulter2(XML_Parser parser,
                                            const XML_Char *context,
                                            const XML_Char *base,
                                            const XML_Char *systemId,
                                            const XML_Char *publicId);

extern int XMLCALL external_entity_unfinished_attlist(XML_Parser parser,
                                                      const XML_Char *context,
                                                      const XML_Char *base,
                                                      const XML_Char *systemId,
                                                      const XML_Char *publicId);

extern int XMLCALL external_entity_handler(XML_Parser parser,
                                           const XML_Char *context,
                                           const XML_Char *base,
                                           const XML_Char *systemId,
                                           const XML_Char *publicId);

extern int XMLCALL external_entity_duff_loader(XML_Parser parser,
                                               const XML_Char *context,
                                               const XML_Char *base,
                                               const XML_Char *systemId,
                                               const XML_Char *publicId);

extern int XMLCALL external_entity_dbl_handler(XML_Parser parser,
                                               const XML_Char *context,
                                               const XML_Char *base,
                                               const XML_Char *systemId,
                                               const XML_Char *publicId);

extern int XMLCALL external_entity_dbl_handler_2(XML_Parser parser,
                                                 const XML_Char *context,
                                                 const XML_Char *base,
                                                 const XML_Char *systemId,
                                                 const XML_Char *publicId);

extern int XMLCALL external_entity_alloc_set_encoding(XML_Parser parser,
                                                      const XML_Char *context,
                                                      const XML_Char *base,
                                                      const XML_Char *systemId,
                                                      const XML_Char *publicId);

extern int XMLCALL external_entity_reallocator(XML_Parser parser,
                                               const XML_Char *context,
                                               const XML_Char *base,
                                               const XML_Char *systemId,
                                               const XML_Char *publicId);

extern int XMLCALL external_entity_alloc(XML_Parser parser,
                                         const XML_Char *context,
                                         const XML_Char *base,
                                         const XML_Char *systemId,
                                         const XML_Char *publicId);

extern int XMLCALL external_entity_parser_create_alloc_fail_handler(
    XML_Parser parser, const XML_Char *context, const XML_Char *base,
    const XML_Char *systemId, const XML_Char *publicId);

struct AccountingTestCase {
  const char *primaryText;
  const char *firstExternalText;  /* often NULL */
  const char *secondExternalText; /* often NULL */
  const unsigned long long expectedCountBytesIndirectExtra;
};

extern int accounting_external_entity_ref_handler(XML_Parser parser,
                                                  const XML_Char *context,
                                                  const XML_Char *base,
                                                  const XML_Char *systemId,
                                                  const XML_Char *publicId);

/* NotStandalone handlers */

extern int XMLCALL reject_not_standalone_handler(void *userData);

extern int XMLCALL accept_not_standalone_handler(void *userData);

/* Attribute List handlers */

typedef struct AttTest {
  const char *definition;
  const XML_Char *element_name;
  const XML_Char *attr_name;
  const XML_Char *attr_type;
  const XML_Char *default_value;
  int is_required;
} AttTest;

extern void XMLCALL verify_attlist_decl_handler(
    void *userData, const XML_Char *element_name, const XML_Char *attr_name,
    const XML_Char *attr_type, const XML_Char *default_value, int is_required);

/* Character data handlers */

extern void XMLCALL clearing_aborting_character_handler(void *userData,
                                                        const XML_Char *s,
                                                        int len);

extern void XMLCALL parser_stop_character_handler(void *userData,
                                                  const XML_Char *s, int len);

extern void XMLCALL cr_cdata_handler(void *userData, const XML_Char *s,
                                     int len);

extern void XMLCALL rsqb_handler(void *userData, const XML_Char *s, int len);

typedef struct ByteTestData {
  int start_element_len;
  int cdata_len;
  int total_string_len;
} ByteTestData;

extern void XMLCALL byte_character_handler(void *userData, const XML_Char *s,
                                           int len);

extern void XMLCALL ext2_accumulate_characters(void *userData,
                                               const XML_Char *s, int len);

/* Handlers that record their `len` arg and a single identifying character */

struct handler_record_entry {
  const char *name;
  int arg;
};
struct handler_record_list {
  int count;
  struct handler_record_entry entries[50]; // arbitrary big-enough max count
};

extern void XMLCALL record_default_handler(void *userData, const XML_Char *s,
                                           int len);

extern void XMLCALL record_cdata_handler(void *userData, const XML_Char *s,
                                         int len);

extern void XMLCALL record_cdata_nodefault_handler(void *userData,
                                                   const XML_Char *s, int len);

extern void XMLCALL record_skip_handler(void *userData,
                                        const XML_Char *entityName,
                                        int is_parameter_entity);

extern void XMLCALL record_element_start_handler(void *userData,
                                                 const XML_Char *name,
                                                 const XML_Char **atts);

extern void XMLCALL record_element_end_handler(void *userData,
                                               const XML_Char *name);

extern const struct handler_record_entry *
_handler_record_get(const struct handler_record_list *storage, int index,
                    const char *file, int line);

#  define handler_record_get(storage, index)                                   \
    _handler_record_get((storage), (index), __FILE__, __LINE__)

#  define assert_record_handler_called(storage, index, expected_name,          \
                                       expected_arg)                           \
    do {                                                                       \
      const struct handler_record_entry *e                                     \
          = handler_record_get(storage, index);                                \
      assert_true(strcmp(e->name, expected_name) == 0);                        \
      assert_true(e->arg == (expected_arg));                                   \
    } while (0)

/* Entity Declaration Handlers */
#  define ENTITY_MATCH_FAIL (-1)
#  define ENTITY_MATCH_NOT_FOUND (0)
#  define ENTITY_MATCH_SUCCESS (1)

extern void XMLCALL param_entity_match_handler(
    void *userData, const XML_Char *entityName, int is_parameter_entity,
    const XML_Char *value, int value_length, const XML_Char *base,
    const XML_Char *systemId, const XML_Char *publicId,
    const XML_Char *notationName);

extern void param_entity_match_init(const XML_Char *name,
                                    const XML_Char *value);

extern int get_param_entity_match_flag(void);

/* Misc handlers */

extern void XMLCALL xml_decl_handler(void *userData, const XML_Char *version,
                                     const XML_Char *encoding, int standalone);

extern void XMLCALL param_check_skip_handler(void *userData,
                                             const XML_Char *entityName,
                                             int is_parameter_entity);

extern void XMLCALL data_check_comment_handler(void *userData,
                                               const XML_Char *data);

extern void XMLCALL selective_aborting_default_handler(void *userData,
                                                       const XML_Char *s,
                                                       int len);

extern void XMLCALL suspending_comment_handler(void *userData,
                                               const XML_Char *data);

extern void XMLCALL element_decl_suspender(void *userData, const XML_Char *name,
                                           XML_Content *model);

extern void XMLCALL suspend_after_element_declaration(void *userData,
                                                      const XML_Char *name,
                                                      XML_Content *model);

extern void XMLCALL accumulate_pi_characters(void *userData,
                                             const XML_Char *target,
                                             const XML_Char *data);

extern void XMLCALL accumulate_comment(void *userData, const XML_Char *data);

extern void XMLCALL accumulate_entity_decl(
    void *userData, const XML_Char *entityName, int is_parameter_entity,
    const XML_Char *value, int value_length, const XML_Char *base,
    const XML_Char *systemId, const XML_Char *publicId,
    const XML_Char *notationName);

extern void XMLCALL accumulate_char_data_and_suspend(void *userData,
                                                     const XML_Char *s,
                                                     int len);

extern void XMLCALL accumulate_start_element(void *userData,
                                             const XML_Char *name,
                                             const XML_Char **atts);

extern void XMLCALL accumulate_characters(void *userData, const XML_Char *s,
                                          int len);

extern void XMLCALL accumulate_attribute(void *userData, const XML_Char *name,
                                         const XML_Char **atts);

extern void XMLCALL ext_accumulate_characters(void *userData, const XML_Char *s,
                                              int len);

typedef struct default_check {
  const XML_Char *expected;
  const int expectedLen;
  XML_Bool seen;
} DefaultCheck;

void XMLCALL checking_default_handler(void *userData, const XML_Char *s,
                                      int len);

typedef struct {
  XML_Parser parser;
  CharData *storage;
} ParserPlusStorage;

extern void XMLCALL
accumulate_and_suspend_comment_handler(void *userData, const XML_Char *data);

#endif /* XML_HANDLERS_H */

#ifdef __cplusplus
}
#endif
