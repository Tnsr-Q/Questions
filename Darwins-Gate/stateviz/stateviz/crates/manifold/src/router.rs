#[derive(Default)]
pub struct Router;
impl Router {
    pub fn mount_namespace(&mut self, _ns: &str) {}
    pub fn unmount_namespace(&mut self, _ns: &str) {}
}
