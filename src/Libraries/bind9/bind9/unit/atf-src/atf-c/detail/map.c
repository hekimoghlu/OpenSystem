/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 22, 2024.
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
#include <errno.h>
#include <stdlib.h>
#include <string.h>

#include "atf-c/error.h"
#include "atf-c/utils.h"

#include "map.h"
#include "sanity.h"

/* ---------------------------------------------------------------------
 * Auxiliary functions.
 * --------------------------------------------------------------------- */

struct map_entry {
    char *m_key;
    void *m_value;
    bool m_managed;
};

static
struct map_entry *
new_entry(const char *key, void *value, bool managed)
{
    struct map_entry *me;

    me = (struct map_entry *)malloc(sizeof(*me));
    if (me != NULL) {
        me->m_key = strdup(key);
        if (me->m_key == NULL) {
            free(me);
            me = NULL;
        } else {
            me->m_value = value;
            me->m_managed = managed;
        }
    }

    return me;
}

/* ---------------------------------------------------------------------
 * The "atf_map_citer" type.
 * --------------------------------------------------------------------- */

/*
 * Getters.
 */

const char *
atf_map_citer_key(const atf_map_citer_t citer)
{
    const struct map_entry *me = citer.m_entry;
    PRE(me != NULL);
    return me->m_key;
}

const void *
atf_map_citer_data(const atf_map_citer_t citer)
{
    const struct map_entry *me = citer.m_entry;
    PRE(me != NULL);
    return me->m_value;
}

atf_map_citer_t
atf_map_citer_next(const atf_map_citer_t citer)
{
    atf_map_citer_t newciter;

    newciter = citer;
    newciter.m_listiter = atf_list_citer_next(citer.m_listiter);
    newciter.m_entry = ((const struct map_entry *)
                        atf_list_citer_data(newciter.m_listiter));

    return newciter;
}

bool
atf_equal_map_citer_map_citer(const atf_map_citer_t i1,
                              const atf_map_citer_t i2)
{
    return i1.m_map == i2.m_map && i1.m_entry == i2.m_entry;
}

/* ---------------------------------------------------------------------
 * The "atf_map_iter" type.
 * --------------------------------------------------------------------- */

/*
 * Getters.
 */

const char *
atf_map_iter_key(const atf_map_iter_t iter)
{
    const struct map_entry *me = iter.m_entry;
    PRE(me != NULL);
    return me->m_key;
}

void *
atf_map_iter_data(const atf_map_iter_t iter)
{
    const struct map_entry *me = iter.m_entry;
    PRE(me != NULL);
    return me->m_value;
}

atf_map_iter_t
atf_map_iter_next(const atf_map_iter_t iter)
{
    atf_map_iter_t newiter;

    newiter = iter;
    newiter.m_listiter = atf_list_iter_next(iter.m_listiter);
    newiter.m_entry = ((struct map_entry *)
                       atf_list_iter_data(newiter.m_listiter));

    return newiter;
}

bool
atf_equal_map_iter_map_iter(const atf_map_iter_t i1,
                            const atf_map_iter_t i2)
{
    return i1.m_map == i2.m_map && i1.m_entry == i2.m_entry;
}

/* ---------------------------------------------------------------------
 * The "atf_map" type.
 * --------------------------------------------------------------------- */

/*
 * Constructors and destructors.
 */

atf_error_t
atf_map_init(atf_map_t *m)
{
    return atf_list_init(&m->m_list);
}

atf_error_t
atf_map_init_charpp(atf_map_t *m, const char *const *array)
{
    atf_error_t err;
    const char *const *ptr = array;

    err = atf_map_init(m);
    if (array != NULL) {
        while (!atf_is_error(err) && *ptr != NULL) {
            const char *key, *value;

            key = *ptr;
            INV(key != NULL);
            ptr++;

            if ((value = *ptr) == NULL) {
                err = atf_libc_error(EINVAL, "List too short; no value for "
                    "key '%s' provided", key);  /* XXX: Not really libc_error */
                break;
            }
            ptr++;

            err = atf_map_insert(m, key, strdup(value), true);
        }
    }

    if (atf_is_error(err))
        atf_map_fini(m);

    return err;
}

void
atf_map_fini(atf_map_t *m)
{
    atf_list_iter_t iter;

    atf_list_for_each(iter, &m->m_list) {
        struct map_entry *me = atf_list_iter_data(iter);

        if (me->m_managed)
            free(me->m_value);
        free(me->m_key);
        free(me);
    }
    atf_list_fini(&m->m_list);
}

/*
 * Getters.
 */

atf_map_iter_t
atf_map_begin(atf_map_t *m)
{
    atf_map_iter_t iter;
    iter.m_map = m;
    iter.m_listiter = atf_list_begin(&m->m_list);
    iter.m_entry = atf_list_iter_data(iter.m_listiter);
    return iter;
}

atf_map_citer_t
atf_map_begin_c(const atf_map_t *m)
{
    atf_map_citer_t citer;
    citer.m_map = m;
    citer.m_listiter = atf_list_begin_c(&m->m_list);
    citer.m_entry = atf_list_citer_data(citer.m_listiter);
    return citer;
}

atf_map_iter_t
atf_map_end(atf_map_t *m)
{
    atf_map_iter_t iter;
    iter.m_map = m;
    iter.m_entry = NULL;
    iter.m_listiter = atf_list_end(&m->m_list);
    return iter;
}

atf_map_citer_t
atf_map_end_c(const atf_map_t *m)
{
    atf_map_citer_t iter;
    iter.m_map = m;
    iter.m_entry = NULL;
    iter.m_listiter = atf_list_end_c(&m->m_list);
    return iter;
}

atf_map_iter_t
atf_map_find(atf_map_t *m, const char *key)
{
    atf_list_iter_t iter;

    atf_list_for_each(iter, &m->m_list) {
        struct map_entry *me = atf_list_iter_data(iter);

        if (strcmp(me->m_key, key) == 0) {
            atf_map_iter_t i;
            i.m_map = m;
            i.m_entry = me;
            i.m_listiter = iter;
            return i;
        }
    }

    return atf_map_end(m);
}

atf_map_citer_t
atf_map_find_c(const atf_map_t *m, const char *key)
{
    atf_list_citer_t iter;

    atf_list_for_each_c(iter, &m->m_list) {
        const struct map_entry *me = atf_list_citer_data(iter);

        if (strcmp(me->m_key, key) == 0) {
            atf_map_citer_t i;
            i.m_map = m;
            i.m_entry = me;
            i.m_listiter = iter;
            return i;
        }
    }

    return atf_map_end_c(m);
}

size_t
atf_map_size(const atf_map_t *m)
{
    return atf_list_size(&m->m_list);
}

char **
atf_map_to_charpp(const atf_map_t *l)
{
    char **array;
    atf_map_citer_t iter;
    size_t i;

    array = malloc(sizeof(char *) * (atf_map_size(l) * 2 + 1));
    if (array == NULL)
        goto out;

    i = 0;
    atf_map_for_each_c(iter, l) {
        array[i] = strdup(atf_map_citer_key(iter));
        if (array[i] == NULL) {
            atf_utils_free_charpp(array);
            array = NULL;
            goto out;
        }

        array[i + 1] = strdup((const char *)atf_map_citer_data(iter));
        if (array[i + 1] == NULL) {
            atf_utils_free_charpp(array);
            array = NULL;
            goto out;
        }

        i += 2;
    }
    array[i] = NULL;

out:
    return array;
}

/*
 * Modifiers.
 */

atf_error_t
atf_map_insert(atf_map_t *m, const char *key, void *value, bool managed)
{
    struct map_entry *me;
    atf_error_t err;
    atf_map_iter_t iter;

    iter = atf_map_find(m, key);
    if (atf_equal_map_iter_map_iter(iter, atf_map_end(m))) {
        me = new_entry(key, value, managed);
        if (me == NULL)
            err = atf_no_memory_error();
        else {
            err = atf_list_append(&m->m_list, me, false);
            if (atf_is_error(err)) {
                if (managed)
                    free(value);
                free(me);
            }
        }
    } else {
        me = iter.m_entry;
        if (me->m_managed)
            free(me->m_value);

        INV(strcmp(me->m_key, key) == 0);
        me->m_value = value;
        me->m_managed = managed;

        err = atf_no_error();
    }

    return err;
}
