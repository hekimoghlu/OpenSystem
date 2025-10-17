/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 15, 2023.
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
// modload_plugin - loader interface for dynamically loaded plugin modules
//
#ifndef _H_MODLOAD_PLUGIN
#define _H_MODLOAD_PLUGIN

#include "modloader.h"


namespace Security {


//
// A LoadablePlugin implements itself as a LoadableBundle
//
class LoadablePlugin : public Plugin, public LoadableBundle {
public:
    LoadablePlugin(const char *path);
    
    void load();
    void unload();
    bool isLoaded() const;

	CSSM_SPI_ModuleLoadFunction load;
	CSSM_SPI_ModuleUnloadFunction unload;
	CSSM_SPI_ModuleAttachFunction attach;
	CSSM_SPI_ModuleDetachFunction detach;
        
private:
	PluginFunctions mFunctions;

    template <class FunctionType>
    void findFunction(FunctionType * &func, const char *name)
    { func = (FunctionType *)lookupSymbol(name); }

    bool allowableModulePath(const char *path);
};


} // end namespace Security


#endif //_H_MODLOAD_PLUGIN
