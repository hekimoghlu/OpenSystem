/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 24, 2025.
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
// modloader.h - CSSM module loader interface
//
// This is a thin abstraction of plugin module loading/handling for CSSM.
// The resulting module ("Plugin") notion is specific to CSSM plugin modules.
// This implementation uses MacOS X bundles.
//
#ifndef _H_MODLOADER
#define _H_MODLOADER

#include <exception>
#include <security_utilities/osxcode.h>
#include "cssmint.h"
#include <map>
#include <string>


namespace Security {


//
// A collection of canonical plugin entry points (aka CSSM module SPI)
//
struct PluginFunctions {
	CSSM_SPI_ModuleLoadFunction *load;
	CSSM_SPI_ModuleUnloadFunction *unload;
	CSSM_SPI_ModuleAttachFunction *attach;
	CSSM_SPI_ModuleDetachFunction *detach;
};


//
// An abstract representation of a loadable plugin.
// Note that "loadable" doesn't mean that actual code loading
// is necessarily happening, but let's just assume it might.
//
class Plugin {
    NOCOPY(Plugin)
public:
    Plugin() { }
    virtual ~Plugin() { }

    virtual void load() = 0;
    virtual void unload() = 0;
    virtual bool isLoaded() const = 0;
	
	virtual CSSM_SPI_ModuleLoadFunction load = 0;
	virtual CSSM_SPI_ModuleUnloadFunction unload = 0;
	virtual CSSM_SPI_ModuleAttachFunction attach = 0;
	virtual CSSM_SPI_ModuleDetachFunction detach = 0;
};


//
// The supervisor class that manages searching and loading.
//
class ModuleLoader {
    NOCOPY(ModuleLoader)
public:
    ModuleLoader();
    
    Plugin *operator () (const string &path);
        
private:
    // the table of all loaded modules
    typedef map<string, Plugin *> PluginTable;
    PluginTable mPlugins;
};



} // end namespace Security


#endif //_H_MODLOADER
