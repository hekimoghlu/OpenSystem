/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 28, 2022.
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
#include "krb5_locl.h"

KRB5_LIB_FUNCTION int KRB5_CALLCONV
krb5_prompter_posix(krb5_context context,
		    void *data,
		    const char *name,
		    const char *banner,
		    int num_prompts,
		    krb5_prompt prompts[])
{
    int i;

    if (name)
	fprintf (stderr, "%s\n", name);
    if (banner)
	fprintf (stderr, "%s\n", banner);
    if (name || banner)
	fflush(stderr);
    for (i = 0; i < num_prompts; ++i) {
	if (prompts[i].hidden) {
	    if (UI_UTIL_read_pw_string(prompts[i].reply->data,
				       (int)prompts[i].reply->length,
				       prompts[i].prompt,
				       0))
	       return 1;
	} else {
	    char *s = prompts[i].reply->data;

	    fputs(prompts[i].prompt, stdout);
	    fflush(stdout);
	    if(fgets(prompts[i].reply->data,
		     (int)prompts[i].reply->length,
		     stdin) == NULL)
		return 1;
	    s[strcspn(s, "\n")] = '\0';
	}
    }
    return 0;
}

KRB5_LIB_FUNCTION int KRB5_CALLCONV
krb5_prompter_print_only(krb5_context context,
			 void *data,
			 const char *name,
			 const char *banner,
			 int num_prompts,
			 krb5_prompt prompts[])
{
    if (name)
	fprintf (stderr, "%s\n", name);
    if (banner)
	fprintf (stderr, "%s\n", banner);
    if (name || banner)
	fflush(stderr);

    if (num_prompts) {
	_krb5_debugx(context, 10, "prompter disabled");
	return 1;
    } else {
	return 0;
    }
}
