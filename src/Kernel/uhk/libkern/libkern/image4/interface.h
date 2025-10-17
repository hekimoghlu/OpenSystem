/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 23, 2024.
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
/*!
 * @header
 * Interface structure for the upward-exported AppleImage4 API.
 *
 * This header relies entirely on transitive inclusion from dlxk.h to satisfy
 * its dependencies.
 */
#ifndef __IMAGE4_DLXK_INTERFACE_H
#define __IMAGE4_DLXK_INTERFACE_H

#if !defined(__IMAGE4_XNU_INDIRECT)
#error "Please include <libkern/image4/dlxk.h> instead of this file"
#endif

__BEGIN_DECLS
OS_ASSUME_NONNULL_BEGIN
OS_ASSUME_PTR_ABI_SINGLE_BEGIN

#pragma mark Macros
#define image4_xnu_dlxk_type(_s) _image4_ ## _s ## _dlxk_t
#define image4_xnu_dlxk_fld(_s) dlxk_ ## _s
#define image4_xnu_dlxk_fld_decl(_s) \
	image4_xnu_dlxk_type(_s) image4_xnu_dlxk_fld(_s)

#pragma mark Types
typedef struct _image4_dlxk_interface {
	image4_struct_version_t dlxk_version;
	image4_xnu_dlxk_fld_decl(coprocessor_host);
	image4_xnu_dlxk_fld_decl(coprocessor_ap);
	image4_xnu_dlxk_fld_decl(coprocessor_ap_local);
	image4_xnu_dlxk_fld_decl(coprocessor_cryptex1);
	image4_xnu_dlxk_fld_decl(coprocessor_sep);
	image4_xnu_dlxk_fld_decl(coprocessor_x86);
	image4_xnu_dlxk_fld_decl(environment_init);
	image4_xnu_dlxk_fld_decl(environment_new);
	image4_xnu_dlxk_fld_decl(environment_set_secure_boot);
	image4_xnu_dlxk_fld_decl(environment_set_callbacks);
	image4_xnu_dlxk_fld_decl(environment_copy_nonce_digest);
	image4_xnu_dlxk_fld_decl(environment_roll_nonce);
	image4_xnu_dlxk_fld_decl(environment_generate_nonce_proposal);
	image4_xnu_dlxk_fld_decl(environment_commit_nonce_proposal);
	image4_xnu_dlxk_fld_decl(environment_get_nonce_handle);
	image4_xnu_dlxk_fld_decl(environment_destroy);
	image4_xnu_dlxk_fld_decl(trust_init);
	image4_xnu_dlxk_fld_decl(trust_new);
	image4_xnu_dlxk_fld_decl(trust_set_payload);
	image4_xnu_dlxk_fld_decl(trust_set_booter);
	image4_xnu_dlxk_fld_decl(trust_record_property_bool);
	image4_xnu_dlxk_fld_decl(trust_record_property_integer);
	image4_xnu_dlxk_fld_decl(trust_record_property_data);
	image4_xnu_dlxk_fld_decl(trust_evaluate);
	image4_xnu_dlxk_fld_decl(trust_destroy);
	image4_xnu_dlxk_fld_decl(trust_evaluation_exec);
	image4_xnu_dlxk_fld_decl(trust_evaluation_preflight);
	image4_xnu_dlxk_fld_decl(trust_evaluation_sign);
	image4_xnu_dlxk_fld_decl(trust_evaluation_boot);
	image4_xnu_dlxk_fld_decl(cs_trap_resolve_handler);
	image4_xnu_dlxk_fld_decl(cs_trap_vector_size);
	image4_xnu_dlxk_fld_decl(trust_evaluation_normalize);
	image4_xnu_dlxk_fld_decl(environment_identify);
	image4_xnu_dlxk_fld_decl(environment_get_digest_info);
	image4_xnu_dlxk_fld_decl(environment_flash);
	image4_xnu_dlxk_fld_decl(coprocessor_resolve_from_manifest);
	image4_xnu_dlxk_fld_decl(coprocessor_bootpc);
	image4_xnu_dlxk_fld_decl(coprocessor_vma2);
	image4_xnu_dlxk_fld_decl(coprocessor_vma3);
#if IMAGE4_API_VERSION >= 20240503
	image4_xnu_dlxk_fld_decl(trust_set_result_buffer);
#endif
} image4_dlxk_interface_t;

OS_ASSUME_PTR_ABI_SINGLE_END
OS_ASSUME_NONNULL_END
__END_DECLS

#endif // __IMAGE4_DLXK_INTERFACE_H
