/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 7, 2024.
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
#ifndef SVPROP_H_
#define SVPROP_H_


FT_BEGIN_HEADER


#define FT_SERVICE_ID_PROPERTIES  "properties"


  typedef FT_Error
  (*FT_Properties_SetFunc)( FT_Module    module,
                            const char*  property_name,
                            const void*  value,
                            FT_Bool      value_is_string );

  typedef FT_Error
  (*FT_Properties_GetFunc)( FT_Module    module,
                            const char*  property_name,
                            void*        value );


  FT_DEFINE_SERVICE( Properties )
  {
    FT_Properties_SetFunc  set_property;
    FT_Properties_GetFunc  get_property;
  };


#define FT_DEFINE_SERVICE_PROPERTIESREC( class_,          \
                                         set_property_,   \
                                         get_property_ )  \
  static const FT_Service_PropertiesRec  class_ =         \
  {                                                       \
    set_property_,                                        \
    get_property_                                         \
  };

  /* */


FT_END_HEADER


#endif /* SVPROP_H_ */


/* END */
