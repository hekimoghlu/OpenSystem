/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 14, 2025.
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

# Include for creating start/stop/restart NCF scripts.

instscripts:: FORCE $(INSTALLBASE)/ap2start.ncf $(INSTALLBASE)/ap2auto.ncf $(INSTALLBASE)/ap2rest.ncf $(INSTALLBASE)/ap2stop.ncf

$(INSTALLBASE)/ap2start.ncf:
	@echo $(DL)# NCF to start Apache 2.x in own address space$(DL)> $@
	@echo $(DL)# Make sure that httpstk is not listening on 80$(DL)>> $@
	@echo $(DL)# httpcloseport 80 /silent$(DL)>> $@
	@echo $(DL)# search add SYS:/$(BASEDIR)$(DL)>> $@
	@echo $(DL)load address space = $(BASEDIR) SYS:/$(BASEDIR)/apache2$(DL)>> $@
	@echo $(DL)# If you have problems with 3rd-party modules try to load in OS space.$(DL)>> $@
	@echo $(DL)# load SYS:/$(BASEDIR)/apache2$(DL)>> $@
	@$(ECHONL)>> $@

$(INSTALLBASE)/ap2auto.ncf:
	@echo $(DL)# NCF to start Apache 2.x in own address space$(DL)> $@
	@echo $(DL)# and let automatically restart in case it crashes$(DL)>> $@
	@echo $(DL)# Make sure that httpstk is not listening on 80$(DL)>> $@
	@echo $(DL)# httpcloseport 80 /silent$(DL)>> $@
	@echo $(DL)# search add SYS:/$(BASEDIR)$(DL)>> $@
	@echo $(DL)restart address space = $(BASEDIR) SYS:/$(BASEDIR)/apache2$(DL)>> $@
	@$(ECHONL)>> $@

$(INSTALLBASE)/ap2rest.ncf:
	@echo $(DL)# NCF to restart Apache 2.x in own address space$(DL)> $@
	@echo $(DL)apache2 restart -p $(BASEDIR)$(DL)>> $@
	@echo $(DL)# If you have loaded Apache2.x in OS space use the line below.$(DL)>> $@
	@echo $(DL)# apache2 restart$(DL)>> $@
	@$(ECHONL)>> $@

$(INSTALLBASE)/ap2stop.ncf:
	@echo $(DL)# NCF to stop Apache 2.x in own address space$(DL)> $@
	@echo $(DL)apache2 shutdown -p $(BASEDIR)$(DL)>> $@
	@echo $(DL)# If you have loaded Apache2.x in OS space use the line below.$(DL)>> $@
	@echo $(DL)# apache2 shutdown$(DL)>> $@
	@$(ECHONL)>> $@

$(INSTALLBASE)/ap2prod.ncf:
	@echo $(DL)# NCF to create a product record for Apache 2.x in product database$(DL)> $@
	@echo $(DL)PRODSYNC DEL APACHE$(VERSION_MAJMIN)$(DL)>> $@
	@echo $(DL)PRODSYNC ADD APACHE$(VERSION_MAJMIN) ProductRecord "$(VERSION_STR)" "Apache $(VERSION_STR) Webserver"$(DL)>> $@
	@$(ECHONL)>> $@

