SUBDIRS = src

help:
	@for i in $(SUBDIRS); do \
		echo "make help in $$i..."; \
		(cd $$i; $(MAKE) $(MFLAGS) $(MYMAKEFLAGS) help); done

html:
	@for i in $(SUBDIRS); do \
		echo "make html in $$i..."; \
		(cd $$i; $(MAKE) $(MFLAGS) $(MYMAKEFLAGS) html); done

html-noplot:
	@for i in $(SUBDIRS); do \
		echo "make html-noplot in $$i..."; \
		(cd $$i; $(MAKE) $(MFLAGS) $(MYMAKEFLAGS) html-noplot); done

html-noapi:
	@for i in $(SUBDIRS); do \
		echo "make html-noapi in $$i..."; \
		(cd $$i; $(MAKE) $(MFLAGS) $(MYMAKEFLAGS) html-noapi); done

html-quick:
	@for i in $(SUBDIRS); do \
		echo "make html-quick in $$i..."; \
		(cd $$i; $(MAKE) $(MFLAGS) $(MYMAKEFLAGS) html-quick); done

clean:
	@for i in $(SUBDIRS); do \
		echo "Clearing in $$i..."; \
		(cd $$i; $(MAKE) $(MFLAGS) $(MYMAKEFLAGS) clean); done

doctest:
	@for i in $(SUBDIRS); do \
		echo "Running doctest in $$i..."; \
		(cd $$i; $(MAKE) $(MFLAGS) $(MYMAKEFLAGS) doctest); done

show:
	@for i in $(SUBDIRS); do \
		echo "Running show in $$i..."; \
		(cd $$i; $(MAKE) $(MFLAGS) $(MYMAKEFLAGS) show); done

livehtml:
	@for i in $(SUBDIRS); do \
		echo "Running show in $$i..."; \
		(cd $$i; $(MAKE) $(MFLAGS) $(MYMAKEFLAGS) livehtml); done
