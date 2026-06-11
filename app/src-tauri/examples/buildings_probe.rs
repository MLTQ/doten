// Print world positions of towers/forts at creation, for minimap bounds calibration.
use source2_demo::prelude::*;
use std::io::BufReader;

#[derive(Default)]
struct Probe {
    seen: u32,
}

#[observer]
#[uses_entities]
impl Probe {
    #[on_entity]
    fn on_entity(&mut self, _ctx: &Context, event: EntityEvents, entity: &Entity) -> ObserverResult {
        if event != EntityEvents::Created {
            return Ok(());
        }
        let class = entity.class().name();
        if !matches!(class, "CDOTA_BaseNPC_Tower" | "CDOTA_BaseNPC_Fort") {
            return Ok(());
        }
        self.seen += 1;
        let cell_x: u16 = try_property!(entity, "CBodyComponent.m_cellX").unwrap_or(0);
        let cell_y: u16 = try_property!(entity, "CBodyComponent.m_cellY").unwrap_or(0);
        let vec_x: f32 = try_property!(entity, "CBodyComponent.m_vecX").unwrap_or(0.0);
        let vec_y: f32 = try_property!(entity, "CBodyComponent.m_vecY").unwrap_or(0.0);
        let team: u8 = try_property!(entity, "m_iTeamNum").unwrap_or(0);
        let x = cell_x as f32 * 128.0 + vec_x - 16384.0;
        let y = cell_y as f32 * 128.0 + vec_y - 16384.0;
        println!("{class} team {team} world ({x:.0}, {y:.0})");
        Ok(())
    }
}

fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).expect("usage: buildings_probe <demofile>");
    let input = BufReader::new(std::fs::File::open(path)?);
    let mut parser = Parser::from_reader(input)?;
    parser.register_observer::<Probe>();
    let _ = parser.run_to_tick(30000); // initial snapshot has all buildings
    Ok(())
}
