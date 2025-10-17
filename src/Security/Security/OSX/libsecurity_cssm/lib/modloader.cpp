/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 10, 2023.
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
//
// cssm module loader interface - MACOS X (CFBundle/DYLD) version.
//
// This file provides a C++-style interface to CFBundles as managed by the CF-style
// system interfaces. The implementation looks a bit, well, hybrid - but the visible
// interfaces are pure C++.
//
#include "modloader.h"
#include "modload_plugin.h"
# include "modload_static.h"


namespace Security {


//
// Pull in functions for built-in plugin modules
//
#define BUILTIN(suffix) \
	extern "C" CSSM_SPI_ModuleLoadFunction CSSM_SPI_ModuleLoad ## suffix; \
	extern "C" CSSM_SPI_ModuleUnloadFunction CSSM_SPI_ModuleUnload ## suffix; \
	extern "C" CSSM_SPI_ModuleAttachFunction CSSM_SPI_ModuleAttach ## suffix; \
	extern "C" CSSM_SPI_ModuleDetachFunction CSSM_SPI_ModuleDetach ## suffix; \
	static const PluginFunctions builtin ## suffix = { \
		CSSM_SPI_ModuleLoad ## suffix, CSSM_SPI_ModuleUnload ## suffix, \
		CSSM_SPI_ModuleAttach ## suffix, CSSM_SPI_ModuleDetach ## suffix \
	};

BUILTIN(__apple_csp)
BUILTIN(__apple_file_dl)
BUILTIN(__apple_cspdl)
BUILTIN(__apple_x509_cl)
BUILTIN(__apple_x509_tp)
BUILTIN(__sd_cspdl)


//
// Construct the canonical ModuleLoader object
//
ModuleLoader::ModuleLoader()
{
#if !defined(NO_BUILTIN_PLUGINS)
    mPlugins["*AppleCSP"] = new StaticPlugin(builtin__apple_csp);
    mPlugins["*AppleDL"] = new StaticPlugin(builtin__apple_file_dl);
    mPlugins["*AppleCSPDL"] = new StaticPlugin(builtin__apple_cspdl);
    mPlugins["*AppleX509CL"] = new StaticPlugin(builtin__apple_x509_cl);
    mPlugins["*AppleX509TP"] = new StaticPlugin(builtin__apple_x509_tp);
    mPlugins["*SDCSPDL"] = new StaticPlugin(builtin__sd_cspdl);
#endif //NO_BUILTIN_PLUGINS
}


//
// "Load" a plugin, given its MDS path. At this layer, we are performing
// a purely physical load operation. No code in the plugin is called.
// If "built-in plugins" are enabled, the moduleTable will come pre-initialized
// with certain paths. Since we consult this table before going to disk, this
// means that we'll pick these up first *as long as the paths match exactly*.
// There is nothing magical in the path strings themselves, other than by
// convention. (The convention is "*NAME", which conveniently does not match
// any actual file path.)
//
Plugin *ModuleLoader::operator () (const string &path)
{
    Plugin * &plugin = mPlugins[path];
    if (!plugin) {
		secinfo("cssm", "ModuleLoader(): creating plugin %s", path.c_str());
        plugin = new LoadablePlugin(path.c_str());
	}
	else {
		secinfo("cssm", "ModuleLoader(): FOUND plugin %s, isLoaded %{BOOL}d", 
			path.c_str(), plugin->isLoaded());
		if(!plugin->isLoaded()) {
			plugin->load();
		}
	}
    return plugin;
}


}	// end namespace Security
