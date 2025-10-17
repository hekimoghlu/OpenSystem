/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 4, 2022.
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
#include <stdlib.h>
#include <string.h>

#include "atf-c/build.h"
#include "atf-c/config.h"
#include "atf-c/error.h"

#include "detail/sanity.h"
#include "detail/text.h"

/* ---------------------------------------------------------------------
 * Auxiliary functions.
 * --------------------------------------------------------------------- */

static
atf_error_t
append_config_var(const char *var, atf_list_t *argv)
{
    atf_error_t err;
    atf_list_t words;

    err = atf_text_split(atf_config_get(var), " ", &words);
    if (atf_is_error(err))
        goto out;

    atf_list_append_list(argv, &words);

out:
    return err;
}

static
atf_error_t
append_arg1(const char *arg, atf_list_t *argv)
{
    return atf_list_append(argv, strdup(arg), true);
}

static
atf_error_t
append_arg2(const char *flag, const char *arg, atf_list_t *argv)
{
    atf_error_t err;

    err = append_arg1(flag, argv);
    if (!atf_is_error(err))
        err = append_arg1(arg, argv);

    return err;
}

static
atf_error_t
append_optargs(const char *const optargs[], atf_list_t *argv)
{
    atf_error_t err;

    err = atf_no_error();
    while (*optargs != NULL && !atf_is_error(err)) {
        err = append_arg1(strdup(*optargs), argv);
        optargs++;
    }

    return err;
}

static
atf_error_t
append_src_out(const char *src, const char *obj, atf_list_t *argv)
{
    atf_error_t err;

    err = append_arg2("-o", obj, argv);
    if (atf_is_error(err))
        goto out;

    err = append_arg1("-c", argv);
    if (atf_is_error(err))
        goto out;

    err = append_arg1(src, argv);

out:
    return err;
}

static
atf_error_t
list_to_array(const atf_list_t *l, char ***ap)
{
    atf_error_t err;
    char **a;

    a = (char **)malloc((atf_list_size(l) + 1) * sizeof(char *));
    if (a == NULL)
        err = atf_no_memory_error();
    else {
        char **aiter;
        atf_list_citer_t liter;

        aiter = a;
        atf_list_for_each_c(liter, l) {
            *aiter = strdup((const char *)atf_list_citer_data(liter));
            aiter++;
        }
        *aiter = NULL;

        err = atf_no_error();
    }
    *ap = a; /* Shut up warnings in the caller about uninitialized *ap. */

    return err;
}

/* ---------------------------------------------------------------------
 * Free functions.
 * --------------------------------------------------------------------- */

atf_error_t
atf_build_c_o(const char *sfile,
              const char *ofile,
              const char *const optargs[],
              char ***argv)
{
    atf_error_t err;
    atf_list_t argv_list;

    err = atf_list_init(&argv_list);
    if (atf_is_error(err))
        goto out;

    err = append_config_var("atf_build_cc", &argv_list);
    if (atf_is_error(err))
        goto out_list;

    err = append_config_var("atf_build_cppflags", &argv_list);
    if (atf_is_error(err))
        goto out_list;

    err = append_config_var("atf_build_cflags", &argv_list);
    if (atf_is_error(err))
        goto out_list;

    if (optargs != NULL) {
        err = append_optargs(optargs, &argv_list);
        if (atf_is_error(err))
            goto out_list;
    }

    err = append_src_out(sfile, ofile, &argv_list);
    if (atf_is_error(err))
        goto out_list;

    err = list_to_array(&argv_list, argv);
    if (atf_is_error(err))
        goto out_list;

out_list:
    atf_list_fini(&argv_list);
out:
    return err;
}

atf_error_t
atf_build_cpp(const char *sfile,
              const char *ofile,
              const char *const optargs[],
              char ***argv)
{
    atf_error_t err;
    atf_list_t argv_list;

    err = atf_list_init(&argv_list);
    if (atf_is_error(err))
        goto out;

    err = append_config_var("atf_build_cpp", &argv_list);
    if (atf_is_error(err))
        goto out_list;

    err = append_config_var("atf_build_cppflags", &argv_list);
    if (atf_is_error(err))
        goto out_list;

    if (optargs != NULL) {
        err = append_optargs(optargs, &argv_list);
        if (atf_is_error(err))
            goto out_list;
    }

    err = append_arg2("-o", ofile, &argv_list);
    if (atf_is_error(err))
        goto out_list;

    err = append_arg1(sfile, &argv_list);
    if (atf_is_error(err))
        goto out_list;

    err = list_to_array(&argv_list, argv);
    if (atf_is_error(err))
        goto out_list;

out_list:
    atf_list_fini(&argv_list);
out:
    return err;
}

atf_error_t
atf_build_cxx_o(const char *sfile,
                const char *ofile,
                const char *const optargs[],
                char ***argv)
{
    atf_error_t err;
    atf_list_t argv_list;

    err = atf_list_init(&argv_list);
    if (atf_is_error(err))
        goto out;

    err = append_config_var("atf_build_cxx", &argv_list);
    if (atf_is_error(err))
        goto out_list;

    err = append_config_var("atf_build_cppflags", &argv_list);
    if (atf_is_error(err))
        goto out_list;

    err = append_config_var("atf_build_cxxflags", &argv_list);
    if (atf_is_error(err))
        goto out_list;

    if (optargs != NULL) {
        err = append_optargs(optargs, &argv_list);
        if (atf_is_error(err))
            goto out_list;
    }

    err = append_src_out(sfile, ofile, &argv_list);
    if (atf_is_error(err))
        goto out_list;

    err = list_to_array(&argv_list, argv);
    if (atf_is_error(err))
        goto out_list;

out_list:
    atf_list_fini(&argv_list);
out:
    return err;
}
