/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 10, 2024.
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
char cvsroot_module_cxx[] = "$Id: module.cxx 10003 2007-10-17 21:42:11Z wsfulton $";

#include "swigmod.h"

struct Module {
  ModuleFactory fac;
  char *name;
  Module *next;
   Module(const char *n, ModuleFactory f) {
    fac = f;
    name = new char[strlen(n) + 1];
     strcpy(name, n);
     next = 0;
  } ~Module() {
    delete[]name;
  }
};

static Module *modules = 0;

/* -----------------------------------------------------------------------------
 * void Swig_register_module()
 *
 * Register a module.
 * ----------------------------------------------------------------------------- */

void Swig_register_module(const char *n, ModuleFactory f) {
  Module *m = new Module(n, f);
  m->next = modules;
  modules = m;
}

/* -----------------------------------------------------------------------------
 * Language *Swig_find_module()
 *
 * Given a command line option, locates the factory function.
 * ----------------------------------------------------------------------------- */

ModuleFactory Swig_find_module(const char *name) {
  Module *m = modules;
  while (m) {
    if (strcmp(m->name, name) == 0) {
      return m->fac;
    }
    m = m->next;
  }
  return 0;
}
