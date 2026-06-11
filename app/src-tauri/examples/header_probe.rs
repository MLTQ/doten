// Print the CDemoFileHeader of a replay (build number, map name, etc).
use source2_demo::prelude::*;
use source2_demo::proto::CDemoFileHeader;
use std::io::BufReader;

#[derive(Default)]
struct HeaderProbe;

#[observer]
impl HeaderProbe {
    #[on_message]
    fn header(&mut self, _ctx: &Context, m: CDemoFileHeader) -> ObserverResult {
        println!("{}", serde_json::to_string_pretty(&m)?);
        Ok(())
    }
}

fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).expect("usage: header_probe <demofile>");
    let input = BufReader::new(std::fs::File::open(path)?);
    let mut parser = Parser::from_reader(input)?;
    parser.register_observer::<HeaderProbe>();
    // header is at the start; a few ticks is plenty
    let _ = parser.run_to_tick(300);
    Ok(())
}
