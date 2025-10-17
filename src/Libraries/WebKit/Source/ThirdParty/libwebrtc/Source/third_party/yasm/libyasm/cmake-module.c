/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 28, 2023.
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
#include <util.h>

#include <libyasm.h>


typedef struct loaded_module {
    const char *keyword;            /* module keyword */
    void *data;                     /* associated data */
} loaded_module;

static HAMT *loaded_modules[] = {NULL, NULL, NULL, NULL, NULL, NULL};

static void
load_module_destroy(/*@only@*/ void *data)
{
    /* do nothing */
}

void *
yasm_load_module(yasm_module_type type, const char *keyword)
{
    if (!loaded_modules[type])
        return NULL;
    return HAMT_search(loaded_modules[type], keyword);
}

void
yasm_register_module(yasm_module_type type, const char *keyword, void *data)
{
    int replace = 1;

    assert(type < sizeof(loaded_modules));

    if (!loaded_modules[type])
        loaded_modules[type] = HAMT_create(0, yasm_internal_error_);

    HAMT_insert(loaded_modules[type], keyword, data, &replace,
                load_module_destroy);
}

typedef struct {
    yasm_module_type type;
    void (*printfunc) (const char *name, const char *keyword);
} list_one_data;

static int
yasm_list_one_module(void *node, void *d)
{
    list_one_data *data = (list_one_data *)d;
    yasm_arch_module *arch;
    yasm_dbgfmt_module *dbgfmt;
    yasm_objfmt_module *objfmt;
    yasm_listfmt_module *listfmt;
    yasm_parser_module *parser;
    yasm_preproc_module *preproc;

    switch (data->type) {
        case YASM_MODULE_ARCH:
            arch = node;
            data->printfunc(arch->name, arch->keyword);
            break;
        case YASM_MODULE_DBGFMT:
            dbgfmt = node;
            data->printfunc(dbgfmt->name, dbgfmt->keyword);
            break;
        case YASM_MODULE_OBJFMT:
            objfmt = node;
            data->printfunc(objfmt->name, objfmt->keyword);
            break;
        case YASM_MODULE_LISTFMT:
            listfmt = node;
            data->printfunc(listfmt->name, listfmt->keyword);
            break;
        case YASM_MODULE_PARSER:
            parser = node;
            data->printfunc(parser->name, parser->keyword);
            break;
        case YASM_MODULE_PREPROC:
            preproc = node;
            data->printfunc(preproc->name, preproc->keyword);
            break;
    }
    return 0;
}

void
yasm_list_modules(yasm_module_type type,
                  void (*printfunc) (const char *name, const char *keyword))
{
    list_one_data data;

    /* Go through available list, and try to load each one */
    if (!loaded_modules[type])
        return;

    data.type = type;
    data.printfunc = printfunc;

    HAMT_traverse(loaded_modules[type], &data, yasm_list_one_module);
}
