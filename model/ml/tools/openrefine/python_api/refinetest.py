from google.refine import refine

class RefineIT(object):
    project_file = None
    project_format = 'text/line-based/*sv'
    project_options = {}
    project = None

    def setUp(self):
        self.server = refine.RefineServer()
        self.refine = refine.Refine(self.server)
        if self.project_file:
            self.project = self.refine.new_project(
                project_file=self.project_file,
                project_format=self.project_format,
                **self.project_options)

    def tearDown(self):
        if self.project:
            self.project.delete()
            self.project = None