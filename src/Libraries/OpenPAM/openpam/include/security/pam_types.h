/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 11, 2023.
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
#ifndef SECURITY_PAM_TYPES_H_INCLUDED
#define SECURITY_PAM_TYPES_H_INCLUDED

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * XSSO 5.1.1
 */
struct pam_message {
	int	 msg_style;
	char	*msg;
};

struct pam_response {
	char	*resp;
	int	 resp_retcode;
};

/*
 * XSSO 5.1.2
 */
struct pam_conv {
	int	(*conv)(int, const struct pam_message **,
	    struct pam_response **, void *);
	void	*appdata_ptr;
};

/*
 * XSSO 5.1.3
 */
struct pam_handle;
typedef struct pam_handle pam_handle_t;

/*
 * Solaris 9
 */
typedef struct pam_repository {
	char	*type;
	void	*scope;
	size_t	 scope_len;
} pam_repository_t;

#ifdef __cplusplus
}
#endif

#endif /* !SECURITY_PAM_TYPES_H_INCLUDED */
