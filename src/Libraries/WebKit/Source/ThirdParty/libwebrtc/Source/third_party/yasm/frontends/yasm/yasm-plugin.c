/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 21, 2022.
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

#include <string.h>

#include "libyasm-stdint.h"
#include "yasm-plugin.h"

#if defined(_MSC_VER)
#include <windows.h>
#elif defined(__GNUC__)
#include <dlfcn.h>
#endif

static void **loaded_plugins = NULL;
static int num_loaded_plugins = 0;

static void *
load_dll(const char *name)
{
#if defined(_MSC_VER)
    return LoadLibrary(name);
#elif defined(__GNUC__)
    return dlopen(name, RTLD_NOW);
#else
    return NULL;
#endif
}

int
load_plugin(const char *name)
{
    char *path;
    void *lib = NULL;
    void (*init_plugin) (void) = NULL;

    /* Load library */

    path = yasm_xmalloc(strlen(name)+10);
#if defined(_MSC_VER)
    strcpy(path, name);
    strcat(path, ".dll");
    lib = load_dll(path);
#elif defined(__GNUC__)
    strcpy(path, "lib");
    strcat(path, name);
    strcat(path, ".so");
    lib = load_dll(path);
    if (!lib) {
        strcpy(path, name);
        strcat(path, ".so");
        lib = load_dll(path);
    }
#endif
    yasm_xfree(path);
    if (!lib)
        lib = load_dll(name);

    if (!lib)
        return 0;       /* Didn't load successfully */

    /* Add to array of loaded plugins */
    loaded_plugins =
        yasm_xrealloc(loaded_plugins, (num_loaded_plugins+1)*sizeof(void *));
    loaded_plugins[num_loaded_plugins] = lib;
    num_loaded_plugins++;

    /* Get yasm_init_plugin() function and run it */

#if defined(_MSC_VER)
    init_plugin =
        (void (*)(void))GetProcAddress((HINSTANCE)lib, "yasm_init_plugin");
#elif defined(__GNUC__)
    init_plugin = (void (*)(void))(uintptr_t)dlsym(lib, "yasm_init_plugin");
#endif

    if (!init_plugin)
        return 0;       /* Didn't load successfully */

    init_plugin();
    return 1;
}

void
unload_plugins(void)
{
    int i;

    if (!loaded_plugins)
        return;

    for (i = 0; i < num_loaded_plugins; i++) {
#if defined(_MSC_VER)
        FreeLibrary((HINSTANCE)loaded_plugins[i]);
#elif defined(__GNUC__)
        dlclose(loaded_plugins[i]);
#endif
    }
    yasm_xfree(loaded_plugins);
    num_loaded_plugins = 0;
}
