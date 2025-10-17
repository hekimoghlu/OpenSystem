/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 11, 2022.
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
 * alloc.h: enumeration of alloc IDs.
 * Used by test_alloc_fail() to test memory allocation failures.
 * Each entry must be on exactly one line, GetAllocId() depends on that.
 */
typedef enum {
    aid_none = 0,
    aid_qf_dirname_start,
    aid_qf_dirname_now,
    aid_qf_namebuf,
    aid_qf_module,
    aid_qf_errmsg,
    aid_qf_pattern,
    aid_qf_efm_fmtstr,
    aid_qf_efm_fmtpart,
    aid_qf_title,
    aid_qf_mef_name,
    aid_qf_qfline,
    aid_qf_qfinfo,
    aid_qf_dirstack,
    aid_qf_multiline_pfx,
    aid_qf_makecmd,
    aid_qf_linebuf,
    aid_tagstack_items,
    aid_tagstack_from,
    aid_tagstack_details,
    aid_sign_getdefined,
    aid_sign_getplaced,
    aid_sign_define_by_name,
    aid_sign_getlist,
    aid_sign_getplaced_dict,
    aid_sign_getplaced_list,
    aid_insert_sign,
    aid_sign_getinfo,
    aid_newbuf_bvars,
    aid_newwin_wvars,
    aid_newtabpage_tvars,
    aid_blob_alloc,
    aid_get_func,
    aid_last
} alloc_id_T;
