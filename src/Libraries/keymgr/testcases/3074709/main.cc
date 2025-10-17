/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 29, 2024.
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

#include <mach-o/dyld.h>
#include <stdio.h>

typedef void (*thrower)();
typedef void (*call_thrower)(thrower);
typedef void (*call_call_thrower)(call_thrower);

int main()
{
  NSObjectFileImage image, image2;
  NSObjectFileImageReturnCode retCode;
  NSModule module, module2;
  NSSymbol sym;
  call_call_thrower func;
  call_thrower do_call_thrower;

  retCode = NSCreateObjectFileImageFromFile ("plugin2", &image2);
  if (retCode != NSObjectFileImageSuccess)
    {
      fprintf (stderr, "failed to load plugin2\n");
      return 1;
    }
  module2 = NSLinkModule(image2, "plugin2", 
			(NSLINKMODULE_OPTION_BINDNOW 
			 | NSLINKMODULE_OPTION_PRIVATE));
  
  sym = NSLookupSymbolInModule (module2, "_do_call_thrower");
  if (sym == NULL)
    {
      fprintf (stderr, "couldn't find `do_call_thrower' in plugin2\n");
      return 1;
    }
  do_call_thrower = (call_thrower) NSAddressOfSymbol (sym);
  
  retCode = NSCreateObjectFileImageFromFile ("plugin", &image);
  if (retCode != NSObjectFileImageSuccess)
    {
      fprintf (stderr, "failed to load plugin\n");
      return 1;
    }
  module = NSLinkModule(image, "plugin", 
			(NSLINKMODULE_OPTION_BINDNOW 
			 | NSLINKMODULE_OPTION_PRIVATE));
  sym = NSLookupSymbolInModule (module, "_func");
  if (sym == NULL)
    {
      fprintf (stderr, "couldn't find `func' in plugin\n");
      return 1;
    }
  func = (call_call_thrower) NSAddressOfSymbol (sym);
  func (do_call_thrower);

  NSDestroyObjectFileImage (image);
  NSUnLinkModule (module, NSUNLINKMODULE_OPTION_RESET_LAZY_REFERENCES);

  NSDestroyObjectFileImage (image2);
  NSUnLinkModule (module2, NSUNLINKMODULE_OPTION_RESET_LAZY_REFERENCES);

  return 0;
}
